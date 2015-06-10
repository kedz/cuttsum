import cuttsum.judgements
import corenlp as cnlp

df = cuttsum.judgements.get_merged_dataframe()
with cnlp.Server(annotators=["tokenize", "ssplit", "pos", "lemma", "ner", "depparse"]) as s:
    text = df.loc[25, "update text"]
    print text
    print s.annotate(text, return_xml=True)
    doc = s.annotate(text)
    

