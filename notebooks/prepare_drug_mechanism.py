import re
import requests
import gzip
import io
from chembl_webresource_client.new_client import new_client
from Bio import Entrez, SeqIO
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import torch
import os
from requests.adapters import HTTPAdapter, Retry
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==== CONFIG ====
Entrez.email = "your_email@example.com"  # Required by NCBI
N_DRUGS = 50  # number of new drug-target pairs to fetch

# ==== STEP 0: Download human reference proteome from UniProt ====
print("Downloading human reference proteome from UniProt...")
url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=(proteome:UP000005640)"
r = requests.get(url)
r.raise_for_status()

gene_to_seq = {}
with gzip.open(io.BytesIO(r.content), "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        header = record.description
        seq = str(record.seq)
        gene = None
        for token in header.split():
            if token.startswith("GN="):
                gene = token.split("=")[1]
                break
        if gene and gene not in gene_to_seq:
            gene_to_seq[gene] = seq

print(f"Loaded {len(gene_to_seq)} human protein sequences from UniProt")

# ==== STEP 0.5: Load existing dataset ====
existing_drugs = set()
if os.path.exists("drug_target_mechanistic_dataset.csv"):
    existing_df = pd.read_csv("drug_target_mechanistic_dataset.csv")
    existing_drugs = set(existing_df['drug_name'].astype(str))
    print(f"Found {len(existing_drugs)} existing drugs in dataset.")

# ==== STEP 1: Summarization model ====
model_id = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==== STEP 2: ChEMBL fetch (skip existing drugs) ====
def fetch_chembl_data(n=5, max_targets=20, existing_drugs=set()):
    molecules_api = new_client.molecule
    targets_api = new_client.target
    activities_api = new_client.activity

    drugs_data = []
    fetched = 0
    page = 0
    page_size = 50  # fetch in pages

    while fetched < n:
        molecules = molecules_api.filter(max_phase=4)[page*page_size:(page+1)*page_size]
        if not molecules:
            print("No more drugs available in ChEMBL.")
            break
        print(f"Scanning page {page+1}, {len(molecules)} molecules")

        for mol in molecules:
            mol_name = mol.get('pref_name', None)
            if not mol_name:
                continue

            # Skip if already in dataset
            if mol_name in existing_drugs:
                continue

            mol_id = mol['molecule_chembl_id']
            mol_struct = mol.get("molecule_structures")
            smiles = None
            if mol_struct and isinstance(mol_struct, dict):
                smiles = mol_struct.get("canonical_smiles", None)
            # Skip drugs without SMILES
            if not smiles:
                continue

            # Get all linked targets
            activities = activities_api.filter(molecule_chembl_id=mol_id).only(['target_chembl_id'])[:max_targets]
            unique_target_ids = list({a['target_chembl_id'] for a in activities if 'target_chembl_id' in a})

            for target_id in unique_target_ids:
                target_info = targets_api.filter(target_chembl_id=target_id).only(['pref_name', 'target_components'])[0]
                target_name = target_info.get('pref_name', None)

                # Extract UniProt accession
                accession = None
                for comp in target_info.get('target_components', []):
                    acc = comp.get('accession', None)
                    if acc:
                        accession = acc
                        break

                # Fetch protein sequence
                aa_seq = None
                if accession:
                    uni_url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
                    try:
                        resp = requests.get(uni_url, timeout=10)
                        if resp.status_code == 200:
                            fasta_lines = resp.text.strip().split("\n")
                            aa_seq = "".join(fasta_lines[1:])
                    except Exception as e:
                        print(f"Error fetching UniProt {accession}: {e}")

                drugs_data.append({
                    "drug_name": mol_name,
                    "drug_smiles": smiles,
                    "target_name": target_name,
                    "target_accession": accession,
                    "target_sequence": aa_seq
                })

            # Mark this drug as processed (even if multiple targets)
            existing_drugs.add(mol_name)
            fetched += 1
            print(f"Added new drug: {mol_name} (total {fetched}/{n})")

            if fetched >= n:
                break
        page += 1

    return drugs_data

# ==== STEP 3: PubMed fetch with retries ====
session = requests.Session()
session.headers.update({"User-Agent": "DrugMechanismPipeline/1.0 (your_email@example.com)"})
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def fetch_pubmed_texts(drug, target, target_accession, max_results=5):
    biokeywords = [
        "mechanism", "action", " pathway", "signaling", "binding", "interaction",
        "association", "docking", "bind", "complex formation", "attach", "affinity",
        "recognition", "occupancy", "inhibition", "suppress", "block", "antagonize",
        "downregulate", "repress", "impair", "interfere", "attenuate", "prevent",
        "decrease activity", "activation", "stimulate", "induce", "enhance",
        "upregulate", "potentiate", "promote", "trigger", "initiate", "agonist",
        "recruitment"
    ]
    base_query = f'"{drug}" AND ({target} OR {target_accession}) AND (' + ' OR '.join(biokeywords) + ')'
    sentences = []

    for db in ["pubmed", "pmc"]:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": db, "term": base_query, "retmax": max_results, "retmode": "json"}

        try:
            search_resp = session.get(url, params=params, timeout=10)
            search_resp.raise_for_status()
            ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"Error searching {db} for {drug}-{target}: {e}")
            continue

        if not ids:
            continue

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {"db": db, "id": ",".join(ids), "retmode": "xml"}

        try:
            fetch_resp = session.get(fetch_url, params=fetch_params, timeout=10)
            fetch_resp.raise_for_status()
        except Exception as e:
            print(f"Error fetching from {db} for {drug}-{target}: {e}")
            continue

        soup = BeautifulSoup(fetch_resp.content, features="xml")
        for abstract in soup.find_all("abstracttext"):
            text = abstract.get_text()
            matched_sentences = [
                sent.strip() for sent in text.split(". ")
                if any(k in sent.lower() for k in biokeywords)
            ]
            sentences.extend(matched_sentences if matched_sentences else [text.strip()])

        for body in soup.find_all("body"):
            text = body.get_text()
            matched_sentences = [
                sent.strip() for sent in text.split(". ")
                if any(k in sent.lower() for k in biokeywords)
            ]
            sentences.extend(matched_sentences)

    return ". ".join(sentences).replace('\n', ' ')

# ==== STEP 4: Summarization ====
def generate_mechanistic_text(drug, target, target_accession, raw_text, max_new_tokens=256):
    prompt = f"""
From biomedical text below, extract, summarize and consolidate a concise mechanistic explanation
of how the drug {drug} interacts with its target {target} ({target_accession}).
Very important: Keep only facts and information from the text below:
{raw_text}
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    pad_id = model.config.pad_token_id or model.config.eos_token_id

    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
        )

    mech_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if (drug and drug.lower() in mech_text.lower()):
        mech_text = re.sub(re.escape(drug), "DRUG", mech_text, flags=re.IGNORECASE)
    if target:
        mech_text = re.sub(re.escape(target), "TARGET", mech_text, flags=re.IGNORECASE)
    if target_accession:
        mech_text = re.sub(re.escape(target_accession), "TARGET", mech_text, flags=re.IGNORECASE)
    return mech_text

# ==== STEP 5: Fetch new drugs ====
data = fetch_chembl_data(n=N_DRUGS, existing_drugs=existing_drugs)

# ==== STEP 6: Main pipeline ====
results = []
for ind, entry in enumerate(tqdm(data, desc="Processing drug-target pairs", unit="pair")):
    start_time = time.perf_counter()

    drug = entry['drug_name']
    target = entry['target_name']
    target_accession = entry['target_accession']

    if entry['target_sequence']:
        raw_text = fetch_pubmed_texts(drug, target, target_accession, max_results=5)
        time.sleep(0.5)
        if drug and (drug.lower() in raw_text.lower()):
            mech_text = generate_mechanistic_text(drug, target, target_accession, raw_text)
        else:
            mech_text = None
    else:
        raw_text = None
        mech_text = None

    results.append({
        "drug_name": drug,
        "drug_smiles": entry['drug_smiles'],
        "target_name": target,
        "target_accession": entry['target_accession'],
        "target_sequence": entry['target_sequence'],
        "raw_literature_text": raw_text,
        "mechanistic_explanation": mech_text
    })

    # Save frequently
    if (ind % 5 == 0) or (ind == len(data) - 1):
        df = pd.DataFrame(results)
        fin_df = df[~df['target_sequence'].isna()]
        final_df = fin_df[~fin_df['mechanistic_explanation'].isna()]

        if os.path.exists("drug_target_mechanistic_dataset.csv"):
            final_df.to_csv("drug_target_mechanistic_dataset.csv", mode='a', header=False, index=False)
        else:
            final_df.to_csv("drug_target_mechanistic_dataset.csv", index=False)

        tqdm.write(f"Appended {len(final_df)} new entries to drug_target_mechanistic_dataset.csv")

    elapsed = time.perf_counter() - start_time
    tqdm.write(f"Iteration {ind+1}/{len(data)} took {elapsed:.2f} seconds")
