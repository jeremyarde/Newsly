from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def things():
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)

    url = "https://www.cbc.ca/news/canada/toronto/skinny-dipping-sharks-ripleys-1.4862945"
    parser = HtmlParser.from_url(url, Tokenizer('english'))

    text = "A man who swam naked among sharks at Ripley's Aquarium of Canada in downtown Toronto late Friday is wanted in a violent assault earlier in the evening, police say. \r\n\r\nSpokesperson Katrina Arrogante said investigators from the city's west-end 14 Division and the downtown 52 Division connected the incidents on Monday through evidence and the clothing he was seen wearing.\r\n\r\nThe assault took place outside Medieval Times at Exhibition Place around 8 p.m. ET Friday, police said, and the victim suffered serious injuries. It's believed the suspect fled and headed to the aquarium, around five kilometres east. Officers were called to the popular tourist attraction two hours later.\r\n\r\n\r\nPolice are looking for this man, described as between 35 and 40, five-foot, 10 inches tall, about 220 pounds, with a heavy build. He has a shaved head, goatee and missing teeth. (Toronto Police Service)\r\nOne minute-long video posted on YouTube shows a man taking off his clothes and diving into the Dangerous Lagoon, a 2.9-million-litre tank that offers an underwater gallery to dozens of marine animals, including 17 sharks.\r\n\r\nThe naked man can be seen doing the breaststroke on the surface of the water while sand tiger sharks swim within centimetres of his feet. \r\n\r\nI was scared I was going to witness the death of this guy.\r\n- Erinn   Acland , witness\r\nGreen sawfish, green sea turtles, green moray eels and other species of tropical fish are also housed in the tank, according to the aquarium's website. \r\n\r\nSecurity at the popular tourist attraction asked the man to leave shortly before 10:30 p.m. ET but he refused, said Jenifferjit Sidhu, a spokesperson for Toronto Police Service.\r\n\r\nInstead, he swam to the edge of the enclosure and emerged from the tank before doing a backward flip into the water, she told CBC Toronto on Monday.\r\n\r\n\r\nCBC News Toronto\r\nWatch a man skinny dip with sharks at Ripley's Aquarium\r\n   WATCH  \r\n00:00 00:55   \r\n\r\n\r\nA nude swimmer dove into the shark tank in Toronto\u2019s Ripley\u2019s Aquarium. No marine animals were harmed, but Toronto Police said the stunt was \"extremely dangerous.\" 0:55\r\nVisitor Erinn Acland said she heard the \"big splash\" and thought the trainers were feeding the sharks. As Acland and her boyfriend approached it, she said, they saw a man in the water. \r\n\r\n\"The guy seemed totally relaxed and there were sharks, like, everywhere,\" she told CBC Toronto. \"He appeared to be totally nude and, like, laughing.\"  \r\n\r\nAcland described the display as unexpected and horrifying.\r\n\r\n\"I don't know what would possess someone to do that. It's totally insane to me,\" she explained. \r\n\r\n\"I was scared I was going to witness the death of this guy.\"\r\n\r\nA video by a visitor who captured the man's aquatic adventure has received more than 5,000 views on YouTube. \r\n\r\nOn-site security called police, said Sidhu."

    x = summarizer(parser.document, 10)

    print('whoop')
