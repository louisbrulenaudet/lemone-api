![Thumbnail](https://github.com/louisbrulenaudet/lemone-api/blob/main/assets/thumbnail.png?raw=true)

# Lemone: the API for French tax data retrieval and classification.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue) ![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg) ![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg) ![Package Manager](https://img.shields.io/badge/package%20manager-uv-purple.svg)

The API is tailored to meet the specific demands of information retrieval and classification across large-scale tax-related corpora, supporting the implementation of production-ready Retrieval-Augmented Generation (RAG) applications. Its primary purpose is to enhance the efficiency and accuracy of legal processes in the french taxation domain, with an emphasis on delivering consistent performance in real-world settings, while also contributing to advancements in legal natural language processing research.

# Usage

This API is based on the use of uv for package management, ruff for linting and type validation and docker (with dramatiq and redis for asynchronous task management) and can be used exclusively on the basis of the Makefile, whose list of commands can be obtained with `make help`.

To launch the application and associated services, you need to run:

```bash
make build
make start
```

Or using Docker:
```bash
docker pull louisbrulenaudet/lemone-api
```

# API Documentation

## Endpoints

The API provides both synchronous and asynchronous endpoints for each operation. Synchronous endpoints return results immediately, while asynchronous endpoints return a task ID that can be used to check the status and retrieve results later.

### 1. Similarity API

**Purpose:** Calculate similarity scores between multiple input texts. This endpoint computes a similarity matrix showing the relationships between provided texts, useful for finding related documents or measuring text similarity. Features built-in caching (60s TTL) and supports both single and batch document comparisons.

**Endpoint:**
```
POST http://localhost:8687/api/v1/similarity
```

**Request Body:**
```json
{
    "input": [
        "Pour les cessions d’actions, de parts de fondateurs ou de parts de bénéficiaires de sociétés par action, autres que celles des personnes morales à prépondérance immobilière, ainsi que pour les parts ou titres de capital souscrits par les clients des établissements mutualistes ou coopératifs, le droit d’enregistrement est fixé à 0,1 %.",
        "La liste des sociétés françaises dont les titres entrent dans le champ de la taxe sur les transactions financières prévue par les dispositions de l'article 235 ter ZD du code général des impôts au 1er décembre 2022 est mise à jour."
    ]
}
```

**Response:**
```json
{
    "model": "louisbrulenaudet/lemone-embed-pro",
    "object": "list",
    "data": {
        "object": "similarity",
        "data": [
            [
                1.0000003576278687,
                0.43992409110069275
            ],
            [
                0.43992409110069275,
                1.0000003576278687
            ]
        ]
    }
}
```

### 2. Classification API

**Purpose:** Classify text inputs into predefined French tax law categories. Uses specialized models to analyze and categorize text according to the 8-category tax law classification scheme. Includes caching for repeated requests and supports both individual and batch classification.

**Endpoint:**
```
POST http://localhost:8687/api/v1/classification
```

**Request Body:**
```json
{
    "input": [
        "Pour les cessions d’actions, de parts de fondateurs ou de parts de bénéficiaires de sociétés par action, autres que celles des personnes morales à prépondérance immobilière, ainsi que pour les parts ou titres de capital souscrits par les clients des établissements mutualistes ou coopératifs, le droit d’enregistrement est fixé à 0,1 %.",
        "La liste des sociétés françaises dont les titres entrent dans le champ de la taxe sur les transactions financières prévue par les dispositions de l'article 235 ter ZD du code général des impôts au 1er décembre 2022 est mise à jour."
    ]
}
```

**Response:**
```json
{
    "model": "louisbrulenaudet/lemone-router-l",
    "object": "list",
    "data": [
        {
            "object": "classification",
            "label": "Patrimoine et enregistrement",
            "score": 0.9997046589851379,
            "index": 0
        },
        {
            "object": "classification",
            "label": "Fiscalité des entreprises",
            "score": 0.9994685053825378,
            "index": 1
        }
    ]
}
```

### 3. Embeddings API

**Purpose:** Generate vector embeddings for input text. Creates high-quality text representations for use in similarity search, semantic analysis, or other machine learning tasks. Optimized for French tax law content with built-in caching and support for different model variants.

**Endpoint:**
```
POST http://localhost:8687/api/v1/embeddings
```

**Request Body:**
```json
{
    "input": "Pour les cessions d’actions, de parts de fondateurs ou de parts de bénéficiaires de sociétés par action, autres que celles des personnes morales à prépondérance immobilière, ainsi que pour les parts ou titres de capital souscrits par les clients des établissements mutualistes ou coopératifs, le droit d’enregistrement est fixé à 0,1 %.",
    "model": "louisbrulenaudet/lemone-embed-pro"
}
```

**Response:**
```json
{
    "model": "louisbrulenaudet/lemone-embed-pro",
    "object": "list",
    "data": [
        {
            "input": "Pour les cessions d’actions, de parts de fondateurs ou de parts de bénéficiaires de sociétés par action, autres que celles des personnes morales à prépondérance immobilière, ainsi que pour les parts ou titres de capital souscrits par les clients des établissements mutualistes ou coopératifs, le droit d’enregistrement est fixé à 0,1 %.",
            "index": 0,
            "object": "embedding",
            "embedding": [
                -0.10794082283973694,
                0.036172136664390564,
                "..."
            ]
        }
    ]
}
```

Note: All synchronous routes provided in this API are also implemented as asynchronous routes. The asynchronous implementations execute operations using a worker and queue system to handle load and ensure scalability.

###  **Endpoint: Get Evaluation Status**

**Purpose:** Monitor the progress of asynchronous operations. This endpoint allows tracking of any async task (embeddings, similarity, or classification) using its task ID. Returns the task's current status, queue position, and processing details. Essential for managing long-running operations and implementing robust error handling.

**Endpoint:**
```
GET http://127.0.0.1:8687/api/v1/task/status/{task_id}
```

#### **Request Parameters**
| Parameter  | Type     | Description  |
|------------|----------|--------------|
| `task_id`  | `string` | The unique identifier of the evaluation task. Example: `"8cdb0dbc-33f1-4120-9e54-86e247101832"`. |

#### **Response Format**
The API returns a JSON object containing evaluation details.

| Key            | Value         |
|---------------|--------------|
| queue_name    | `{queue_name}` |
| task_name     | `{task_name}`  |
| task_id       | `{task_id}`    |
| task_timestamp | `{task_timestamp}` |

##### **Example Response**
```json
{
	"queue_name": "embeddings",
	"task_name": "embeddings",
	"task_id": "3ec2b643-4d23-4803-9eb0-f5b5c2989d69",
	"task_timestamp": 1738745812497
}
```

## Models

### Lemone-Embed: A Series of Fine-Tuned Embedding Models for French Tax

These series are made up of 7 models: 3 basic models of different sizes trained for 1 epoch, 3 models trained for 2 epochs forming the Boost series, and Pro models with non-RoBERTa architectures.

These sentence transformer models, specifically designed for French taxation, have been fine-tuned on datasets comprising 43 million tokens, integrating blends of semi-synthetic and fully synthetic data generated by GPT-4 Turbo and Llama 3.1 70B. These datasets have been further refined through evol-instruction tuning and manual curation.

#### Training Hardware
- **On Cloud**: No
- **GPU Model**: 1 x NVIDIA H100 NVL
- **CPU Model**: AMD EPYC 9V84 96-Core Processor
- **RAM Size**: 314.68 GB

### Lemone-Router: A Series of Fine-Tuned Classification Models for French Tax

Lemone-router is a series of classification models designed to produce an optimal multi-agent system for different branches of tax law. Trained on a base of 49k lines comprising a set of synthetic questions generated by GPT-4 Turbo and Llama 3.1 70B, which have been further refined through evol-instruction tuning and manual curation and authority documents, these models are based on an 8-category decomposition of the classification scheme derived from the Bulletin officiel des finances publiques - impôts :

```python
label2id = {
    "Bénéfices professionnels": 0,
    "Contrôle et contentieux": 1,
    "Dispositifs transversaux": 2,
    "Fiscalité des entreprises": 3,
    "Patrimoine et enregistrement": 4,
    "Revenus particuliers": 5,
    "Revenus patrimoniaux": 6,
    "Taxes sur la consommation": 7
}

id2label = {
    0: "Bénéfices professionnels",
    1: "Contrôle et contentieux",
    2: "Dispositifs transversaux",
    3: "Fiscalité des entreprises",
    4: "Patrimoine et enregistrement",
    5: "Revenus particuliers",
    6: "Revenus patrimoniaux",
    7: "Taxes sur la consommation"
}
```

This model is a fine-tuned version of [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large).
It achieves the following results on the evaluation set:
- Loss: 0.4734
- Accuracy: 0.9191

#### Training Hardware
- **On Cloud**: No
- **GPU Model**: 1 x NVIDIA H100 NVL
- **CPU Model**: AMD EPYC 9V84 96-Core Processor

## Citation

### BibTeX

If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2025,
  author =       {Louis Brulé Naudet},
  title =        {Lemone: the API for French tax data retrieval and classification.},
  year =         {2025}
  howpublished = {\url{https://huggingface.co/datasets/louisbrulenaudet/lemone-api}},
}
```

```BibTeX
@misc{louisbrulenaudet2024,
  author =       {Louis Brulé Naudet},
  title =        {Lemone-Embed: A Series of Fine-Tuned Embedding Models for French Taxation},
  year =         {2024}
  howpublished = {\url{https://huggingface.co/datasets/louisbrulenaudet/lemone-embed-pro}},
}
```

```BibTeX
@misc{louisbrulenaudet2024,
  author =       {Louis Brulé Naudet},
  title =        {Lemone-Router: A Series of Fine-Tuned Classification Models for French Taxation},
  year =         {2024}
  howpublished = {\url{https://huggingface.co/datasets/louisbrulenaudet/lemone-router-l}},
}
```

## Feedback

If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).
