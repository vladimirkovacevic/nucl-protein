import datasets
import torch
import torch.nn.functional as F
from tqdm import tqdm

from crossmod.model_registry import ModelRegistry


class EmbeddingCache(torch.utils.data.Dataset):
    """This class will precompute embeddings for the data and cache
    them for future reuse."""

    def __init__(
        self,
        data: datasets.Dataset,
        key: str,
        input_ids_name: str,
        attention_mask_name: str,
        emb_model_name: str,
        device: torch.device,
    ) -> None:
        """
        Args:
            data: Huggingface dataset that will be cached.
            key: Column with unique values for each sample (e.g. sequence hash).
            input_ids_name: Name of the input_ids in dataset for the selected modality (e.g. dna_input_ids).
            attention_mask_name: Name of the attention_mask in dataset for the selected modality (e.g. dna_attention_mask).
            emb_model_name: Unique huggingface identifier for the model.
            device: Device on which embeddings will reside.
        """
        self.device = device
        self.key = key
        self.emb_model_name = emb_model_name
        self.emb_model = ModelRegistry.get_model(emb_model_name)
        self.data = self._filter_duplicates(data, key)
        self.cached_embeddings = self._cache_embeddings(
            input_ids_name, attention_mask_name
        )

    def _filter_duplicates(self, data: datasets.Dataset, key: str):
        seen = set()
        filtered_dataset = data.filter(
            lambda example: not (example[key] in seen or seen.add(example[key]))
        )
        return filtered_dataset

    def _cache_embeddings(self, input_ids_name: str, attention_mask_name: str):
        embeddings_cache = {}
        for sample in tqdm(self.data):
            id = sample[self.key]
            inputs = {
                "input_ids": torch.tensor(sample[input_ids_name])
                .unsqueeze(0)
                .to(self.device),
                "attention_mask": torch.tensor(sample[attention_mask_name])
                .unsqueeze(0)
                .to(self.device),
            }
            embeddings_cache[id] = self._compute_embedding(inputs)
        return embeddings_cache

    def _compute_embedding(self, inputs: dict[str, torch.Tensor]):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                embedding = self.emb_model(**inputs).last_hidden_state.detach().cpu()
        return embedding

    def __getitem__(self, keys: list[int]):
        embeddings = []
        for key in keys:
            embeddings.append(self.cached_embeddings[key.item()])

        # pad with 0 to the end
        max_len = max(e.shape[1] for e in embeddings)
        for i in range(len(embeddings)):
            padding_size = max_len - embeddings[i].shape[1]
            if padding_size > 0:
                embeddings[i] = F.pad(embeddings[i], (0, 0, 0, padding_size))
            embeddings[i] = embeddings[i].squeeze(0)

        stacked_embeddings = torch.stack(embeddings, dim=0)
        return stacked_embeddings

    def __len__(self):
        return len(self.data)
