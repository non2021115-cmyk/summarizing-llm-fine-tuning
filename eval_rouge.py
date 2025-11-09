from rouge_score import rouge_scorer
reference_text = """
But here’s what’s valuable for you to notice. At each point in her journey, she had to adapt and grow. During her master’s program, Maria learned how to process complex imaging data. At her first job in a neuroscience startup, she learned to turn those data pipelines into real diagnostic tools used by hospitals. And at DeepMind, she learned to bridge research and product, turning AI models into solutions that improve patient care. The ideas we’ll cover today directly link to your coursework. You’re studying machine learning and data visualization Maria applies those same principles to analyze brain scans and detect early signs of Alzheimer’s disease.
"""
generated_summary = "Maria learned how to process complex imaging data. at her first job in a neuroscience startup, she learned to bridge research and product, turning AI models into solutions that improve patient care."

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_text, generated_summary)

print(scores)