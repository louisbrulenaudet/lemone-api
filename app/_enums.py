import enum

__all__ = [
    "DeviceTypes",
    "ClassificationModels",
    "ClassificationLabels",
    "EmbeddingBackend",
    "EncodingFormats",
    "ErrorCodes",
    "Models",
    "ObjectTypes",
    "SimilarityFunctions",
    "TaskNames",
    "QueueNames",
    "WorkerMaxRetries",
]


class DeviceTypes(enum.StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class EmbeddingBackend(enum.StrEnum):
    TORCH = "torch"
    ONNX = "onnx"
    OPENVINO = "openvino"


class EncodingFormats(enum.StrEnum):
    BINARY = "binary"
    FLOAT32 = "float32"
    INT8 = "int8"
    UINT8 = "uint8"


class ErrorCodes(enum.StrEnum):
    CLASSIFICATION_COMPUTE_ERROR = "CLASSIFICATION_COMPUTE_ERROR"
    EMBEDDING_COMPUTE_ERROR = "EMBEDDING_COMPUTE_ERROR"
    MODEL_REGISTRY_NOT_FOUND = "MODEL_REGISTRY_NOT_FOUND"
    MODEL_NOT_EXIST = "MODEL_NOT_EXIST"
    INVALID_INPUT = "INVALID_INPUT"
    SIMILARITY_COMPUTE_ERROR = "SIMILARITY_COMPUTE_ERROR"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_INITIALIZATION_ERROR = "TASK_INITIALIZATION_ERROR"
    TASK_TRACKING_ERROR = "TASK_TRACKING_ERROR"
    BROKER_INITIALIZATION_ERROR = "BROKER_INITIALIZATION_ERROR"
    BROKER_RESULT_BACKEND_INITIALIZATION_ERROR = (
        "BROKER_RESULT_BACKEND_INITIALIZATION_ERROR"
    )


class Models(enum.StrEnum):
    LEMONE_EMBED_L = "louisbrulenaudet/lemone-embed-l"
    LEMONE_EMBED_L_BOOST = "louisbrulenaudet/lemone-embed-l-boost"
    LEMONE_EMBED_M = "louisbrulenaudet/lemone-embed-m"
    LEMONE_EMBED_M_BOOST = "louisbrulenaudet/lemone-embed-m-boost"
    LEMONE_EMBED_PRO = "louisbrulenaudet/lemone-embed-pro"
    LEMONE_EMBED_S = "louisbrulenaudet/lemone-embed-s"
    LEMONE_EMBED_S_BOOST = "louisbrulenaudet/lemone-embed-s-boost"


class ClassificationModels(enum.StrEnum):
    LEMONE_ROUTER_L = "louisbrulenaudet/lemone-router-l"
    LEMONE_ROUTER_M = "louisbrulenaudet/lemone-router-m"
    LEMONE_ROUTER_S = "louisbrulenaudet/lemone-router-s"


class ClassificationLabels(enum.StrEnum):
    BENEFICES_PROFESSIONNELS = "Bénéfices professionnels"
    CONTROLE_ET_CONTENTIEUX = "Contrôle et contentieux"
    DISPOSITIFS_TRANSVERSAUX = "Dispositifs transversaux"
    FISCALITE_DES_ENTREPRISES = "Fiscalité des entreprises"
    PATRIMOINE_ET_ENREGISTREMENT = "Patrimoine et enregistrement"
    REVENUS_PARTICULIERS = "Revenus particuliers"
    REVENUS_PATRIMONIAUX = "Revenus patrimoniaux"
    TAXES_SUR_LA_CONSOMMATION = "Taxes sur la consommation"


class ObjectTypes(enum.StrEnum):
    EMBEDDING = "embedding"
    SIMILARITY = "similarity"
    CLASSIFICATION = "classification"
    LIST = "list"


class SimilarityFunctions(enum.StrEnum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class TaskNames(enum.StrEnum):
    EMBEDDING = "embedding"
    SIMILARITY = "similarity"
    CLASSIFICATION = "classification"


class TaskStates(enum.StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"


class QueueNames(enum.StrEnum):
    EMBEDDING = "embedding"
    SIMILARITY = "similarity"
    CLASSIFICATION = "classification"


class WorkerMaxRetries(enum.IntEnum):
    EMBEDDING = 3
    SIMILARITY = 3
    CLASSIFICATION = 3
