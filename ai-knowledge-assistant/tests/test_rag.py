import unittest
from rag import chunk_text, prepare_documents, retrieve, grounded_answer

class RagTests(unittest.TestCase):
    def test_chunks_overlap(self):
        self.assertEqual(chunk_text("abcdefghij", 6, 2), ["abcdef", "efghij"])
    def test_empty(self):
        self.assertEqual(chunk_text(" "), [])
    def test_invalid_chunking(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", 4, 4)
    def test_relevant_result(self):
        docs = prepare_documents({"a.txt": "Python pandas machine learning", "b.txt": "Cooking pasta recipes"})
        self.assertEqual(retrieve("Python machine learning", docs)[0]["source"], "a.txt")
    def test_no_match(self):
        self.assertEqual(retrieve("zzzz", prepare_documents({"a": "python code"})), [])
    def test_limit(self):
        docs = prepare_documents({"a": "python python", "b": "python scripts"})
        self.assertEqual(len(retrieve("python", docs, limit=1)), 1)
    def test_no_evidence(self):
        self.assertIn("No relevant", grounded_answer("anything", [], None))
    def test_citations_in_prompt(self):
        class Client:
            class responses:
                @staticmethod
                def create(**kwargs):
                    assert "Source: file.txt" in kwargs["input"]
                    assert "Cite passages" in kwargs["instructions"]
                    return type("Response", (), {"output_text": "Result [1]"})()
        self.assertEqual(grounded_answer("question", [{"source":"file.txt","chunk":1,"text":"evidence"}], Client()), "Result [1]")

if __name__ == "__main__":
    unittest.main()
