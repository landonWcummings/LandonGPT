Code for training and hosting a large language model.

This version is using the Llama 3.2  3 Billion parameter instruct model.

## Data preperation
The first step of this is preparing the data. For me I used my iMessage data. To extract all the raw iMessage data I first used my [iMessage analysis app](https://github.com/landonWcummings/imessageanalysisapp). One of the byproducts of making the macOS app was extracting all my iMessage data and formatting it cleanly. After running the app I had all the raw data cleanly made in Documents>ImessageAnalysisApp>processed_messages.csv  
From here I ran the dataprocess.py file to filter it and organize it into the jsonl file format. There are a few notable and adjustable features of this dataprocess program. 
1 - It only looks at all messages after January 1, 2023. I want the model to mimic my modern self, not my middle school self.
2 - It only looks at direct messages (no group chats included). While there is good data in group chats it would be very hard to use other conversations as input cotext. DMs are much easier as very direct input:output conversationally.
3 - I only used conversation data from my 20 most contacted contacts. I did this in order to better mimic how I text with my close friends
4 - Converted all text to lowercase. My capitilization when texting is strange (if I even capitalize stuff at all) and there is no apparent pattern so capitilization would just confuse the model
5 - Remove all images completely and replaced links with "link". You can't pass images to this LLM and also long URLs would not help the model at all and could confuse it
6 - Removed all repetitive and spammy text. Any time there were more than 3 characters in a row I truncated it to just 3 characters. For example, "oooooooooh" -> "oooh"
7 - Removed all iMessage reactions. For example, '{contact} laughed at "random text"' got removed
8 - Removed all emojis. I trained 8+ models trying to get this to work but ultimately this model wasn't smart/large enough to use them intelligently
9 - Combined all consecutive messages sent by a contact or yourself. Seperated these messages with "|" symbol. For example, if I sent "Haha" followed by "that is amazing" it would now become "Haha | that is amazing"
10 - If one of my contacts sent me message(s) and I replied within 30 minutes it was properly formatted to jsonl. For example, if John sent me "Want to grab dinner tonight?" and I responded with "Yeah | Bartaco?" then it would turn into
{ "input": "Want to grab dinner tonight?", "output": "Yeah | Bartaco?" }

Finally, I wanted to include some information about myself so I had chatGPT generate 80 synthetic conversation style jsonl that used this data
knowyourself = {
   "height": "5 feet 10 inches",
   "age" : "omitted for privacy",
   "future profession": "engineer or entrepreneur",
   "gender": "male",
   "name": "Landon Cummings",
   "political party" : "neither | i am socially left and fiscally right",
   "presidential candidate" : "im impartial | both have major weaknesses",
   "school" : "omitted for privacy",
   "hometown" : "atlanta, georgia"
}
I passed these 80 synthetic conversation facts about me 7+ times into the dataset so it could actually learn it


## Training
Used 4 bit quantization in order to speed up training and inference (you lose some intelligence with this decision). 
I tried everywhere from 2-15 epochs (number of run throughs of the training data) and eventually settled on 3. The more epochs you had the poorer the response quality got to novel questions. Model was overfitting with too many epochs. 
In the first few models I trained I basically just did pretraining where you basically train it to predict and understand english (but it was my messy iMessages). This was not what I wanted to do and the responses were terrible. Eventually I learned that I had to mask the loss values of the input. This allowed me to train a model that responded properly to inputs. View lines 70-80 to see this process in trainer.py
I also had a lot of trouble with my finishing token. For some reason or another I never got the classic < /s> end token or even a special end token to register. Eventually after messing a ton with this I just put the phrase "DONE" at the end of the output and the model learned it perfectly.



## Inference
This uses flask to create a api and ngrok to make it easily publicly available to the web. This is the link if you want to call requests to the model independently https://aeed-64-149-153-183.ngrok-free.app  -note just visiting this site doesn't work and you have to do a command line prompt like :
windows:
curl -X POST "https://aeed-64-149-153-183.ngrok-free.app/generate" ^
-H "Content-Type: application/json" ^
-d "{\"text\": \"tell me who you are?\"}"

mac:
curl -X POST "https://aeed-64-149-153-183.ngrok-free.app/generate" \
-H "Content-Type: application/json" \
-d '{"text": "hello, how are you?"}'

If you just copy and paste these into your command prompt it call the model and you should get a response (in JSON format)



## Final thoughts
This model is very small and therefore pretty stupid. It will say very wrong things and hallucinate a lot. What this model does well is mimic my tone and my general format of texting. You may notice the abreviations I use and utilizing () a lot in my texting.
I think all the synthetic conversations about who I am likely hurt the model inteligence wise but it makes it know my data which is quite important (I'm not jumping into RAG just yet).
It would be cool to try this on a smarter model but that would take more VRAM and much more time.