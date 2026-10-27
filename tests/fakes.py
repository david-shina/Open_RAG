class FakeMessage:
    def __init__(self, content="Answer [S1].", usage=None):
        self.content = content
        self.usage_metadata = usage or {
            "input_tokens": 12,
            "output_tokens": 6,
            "total_tokens": 18,
        }
        self.response_metadata = {}


class FakeChatModel:
    def __init__(self, content="Answer [S1]."):
        self.content = content
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return FakeMessage(self.content)


class FakeGrade:
    def __init__(self, binary_score):
        self.binary_score = binary_score


class FakeGrader:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        score = self.scores.pop(0) if self.scores else "yes"
        return FakeGrade(score)


class FakeTranslator:
    def __init__(self, queries=None):
        self.queries = queries or ["What is the topic?", "Give me the background."]

    def translate(self, query, mode):
        return list(self.queries)


class FakeRetriever:
    def __init__(self, documents=None):
        self.documents = list(documents or [])
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return list(self.documents)


class FakeEmbeddingModel:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
