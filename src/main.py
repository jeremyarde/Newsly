from src.Enums.SummarizerEnums import Summarizer
from src.Summarizers.BaseSummarizer import BaseSummarizer
from src.Summarizers.SumySummarizer import SumySummarizer

sumy = SumySummarizer(summarizerType=Summarizer.LSA)

url = "https://www.cbc.ca/news/canada/toronto/skinny-dipping-sharks-ripleys-1.4862945"
url2 = "https://www.bbc.com/news/business-45986510"

results = sumy.get_summary(url2)
print(results)
