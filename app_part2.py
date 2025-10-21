# %%write app.py
# prompt: import openai, huggingface client and gradio

import openai
import gradio
import huggingface_hub
# %%write -a app.py
# prompt: create a new class Pluto_Happy and name it monty

monty = Pluto_Happy('Monty, Monty Said!')
# %%write app.py
# Prompt: None
# replace the "getenv()" with your key string

import os
monty._openai_key='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'
monty._github_key=os.getenv('github_key')
monty._huggingface_key='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'
monty._kaggle_key='07f17f4937c62103507b2ef27a4b3846'

# %%write -a app.py

# fname = 'toxic_data.csv'
# monty.df_toxic_data = pandas.read_csv(fname)
# %%writefile -a app.py
# prompt: (combine of many seperate prompts and copy code into one code cell)

# for openai version 1.3.8
@add_method(Pluto_Happy)
#
def _fetch_moderate_engine(self):
  self.ai_client = openai.OpenAI(api_key=self._openai_key)
  self.text_model = "text-moderation-latest"
  return
#
@add_method(Pluto_Happy)
# f
def _censor_me(self, p, safer=0.0005):
  self._fetch_moderate_engine()
  resp_orig = self.ai_client.moderations.create(input=p, model=self.text_model)
  resp_dict = resp_orig.model_dump()
  #
  v1 = resp_dict["results"][0]["category_scores"]
  v1 = {key: value if value is not None else 0 for key, value in v1.items()}
  print(f'resp_dic: {resp_dict}')
  print(f'v1: {v1}')

  max_key = max(v1, key=v1.get)
  max_value = v1[max_key]
  sum_value = sum(v1.values())
  #
  v1["is_safer_flagged"] = False
  if (max_value >= safer):
    v1["is_safer_flagged"] = True
  v1["is_flagged"] = resp_dict["results"][0]["flagged"]
  v1['max_key'] = max_key
  v1['max_value'] = max_value
  v1['sum_value'] = sum_value
  v1['safer_value'] = safer
  v1['message'] = p
  return v1
#
@add_method(Pluto_Happy)
def _draw_censor(self,data):
  self._color_mid_gray = '#6c757d'
  exp = (0.01, 0.01)
  x = [data['max_value'], (1-data['max_value'])]
  title=f"\nUnsafe: {data['max_key']}: {(data['max_value']*100):.2f}% Confidence\n"
  lab = [data['max_key'], 'Other 13 categories']
  if (data['is_flagged']):
    col=[self.color_danger, self.color_mid_gray]
  elif (data['is_safer_flagged']):
    col=[self.color_warning, self.color_mid_gray]
    lab = ['Relative Score:\n'+data['max_key'], 'Other 13 categories']
    title=f"\nPersonal Unsafe: {data['max_key']}: {(data['max_value']*100):.2f}% Confidence\n"
  else:
    col=[self.color_mid_gray, self.color_success]
    lab = ['False Negative:\n'+data['max_key'], 'Other 13 categories']
    title='\nSafe Message\n'
  canvas = self._draw_donut(x, lab, col, exp,title)
  return canvas
#
@add_method(Pluto_Happy)
def _draw_donut(self,data,labels,col, exp,title):
  # col = [self.color_danger, self._color_secondary]
  # exp = (0.01, 0.01)
  # Create a pie chart
  canvas, pic = matplotlib.pyplot.subplots()
  pic.pie(data, explode=exp,
    labels=labels,
    colors=col,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color':'#0a0a0a'})
  # Draw a circle at the center of pie to make it look like a donut
  # centre_circle = matplotlib.pyplot.Circle((0,0),0.45,fc='white')
  centre_circle = matplotlib.pyplot.Circle((0,0),0.45,fc=col[0],linewidth=2, ec='white')
  canvas = matplotlib.pyplot.gcf()
  canvas.gca().add_artist(centre_circle)

  # Equal aspect ratio ensures that pie is drawn as a circle.
  pic.axis('equal')
  pic.set_title(title)
  canvas.tight_layout()
  # canvas.show()
  return canvas
#
@add_method(Pluto_Happy)
# def censor_me(self, msg, safer=0.02, ibutton_1=0):
def fetch_toxicity_level(self, msg, safer):
  # safer=0.2
  yjson = self._censor_me(msg,safer)
  _canvas = self._draw_censor(yjson)
  _yjson = json.dumps(yjson, indent=4)
  return (_canvas, _yjson)
  #return(_canvas)
# %%write -a app.py
# prompt: result from a lot of prompt AI and old fashion try and error

import random
def say_hello(val):
  return f"Hello: {val}"
def say_toxic():
  return f"I am toxic"
def fetch_toxic_tweets(maxi=2):
    sample_df = monty.df_toxic_data.sample(maxi)
    is_true = random.choice([True, False])
    c1 = "more_toxic"
    if is_true:
      c1 = "less_toxic"
    toxic1 = sample_df[c1].iloc[0]
    # toxic1 = "cat eats my homework."
    return sample_df.to_html(index=False), toxic1
#
# define all gradio widget/components outside the block for easy to visualize the blocks structure
#
in1 = gradio.Textbox(lines=3, label="Enter Text:")
in2 = gradio.Slider(0.005, .1, value=0.02, step=.005,label="Personalize Safer Value: (larger value is less safe)")
out1 = gradio.Plot(label="Output:")
out2 = gradio.HTML(label="Real-world Toxic Posts/Tweets: *WARNING")
out3 = gradio.Textbox(lines=5, label="Output JSON:")
but1 = gradio.Button("Measure 14 Toxicity", variant="primary",size="sm")
but2 = gradio.Button("Fetch Toxic Text", variant="stop", size="sm")
#
txt1 = """
# Welcome To The Friendly Text Moderation

### Identify 14 categories of text toxicity.

> This NLP (Natural Language Processing) AI demonstration aims to prevent profanity, vulgarity, hate speech, violence, sexism, and other offensive language.
>It is **not an act of censorship**, as the final UI (User Interface) will give the reader, but not a young reader, the option to click on a label to read the toxic message.
>The goal is to create a safer and more respectful environment for you, your colleages, and your family.
> This NLP app is 1 of 3 hands-on courses, ["AI Solution Architect," from ELVTR and Duc Haba](https://elvtr.com/course/ai-solution-architect?utm_source=instructor&utm_campaign=AISA&utm_content=linkedin).
---
### Helpful Instruction:

1. Enter your [harmful] message in the input box.

2. Click the "Measure 14 Toxicity" button.
3. View the result on the Donut plot.
4. (**Optional**) Click on the "Fetch Real World Toxic Dataset" below.
5. There are additional options and notes below.
"""
txt2 = """
## Author and Developer Notes:
---
- The demo uses the cutting-edge (2024) AI Natural Language Processing (NLP) model from OpenAI.
- This NLP app is 1 of 3 hands-on apps from the ["AI Solution Architect," from ELVTR and Duc Haba](https://elvtr.com/course/ai-solution-architect?utm_source=instructor&utm_campaign=AISA&utm_content=linkedin).

- It is not a Generative (GenAI) model, such as Google Gemini or GPT-4.
- The NLP understands the message context, nuance, innuendo, and not just swear words.
- We **challenge you** to trick it, i.e., write a toxic tweet or post, but our AI thinks it is safe. If you win, please send us your message.
- The 14 toxicity categories are as follows:

    1. harassment
    2. harassment threatening
    3. harassment instructions
    4. hate
    5. hate threatening
    6. hate instructions
    7. self harm
    8. self harm instructions
    9. self harm intent
    10. self harm minor
    11. sexual
    12. sexual minors
    13. violence
    14. violence graphic

- If the NLP model classifies the message as "safe," you can still limit the level of toxicity by using the "Personal Safe" slider.
- The smaller the personal-safe value, the stricter the limitation. It means that if you're a young or sensitive adult, you should choose a lower personal-safe value, less than 0.02, to ensure you're not exposed to harmful content.
- The color of the donut plot is as follows:
  - Red is an "unsafe" message by the NLP model
  - Green is a "safe" message
  - Yellow is an "unsafe" message by your toxicity level

- The **"confidence"** score refers to the confidence level in detecting a particular type of toxicity among the 14 tracked types. For instance, if the confidence score is 90%, it indicates a 90% chance that the toxicity detected is of that particular type. In comparison, the remaining 13 toxicities collectively have a 10% chance of being the detected toxicity. Conversely, if the confidence score is 3%, it could indicate any toxicity. It's worth noting that the Red, Green, or Yellow safety levels do not influence the confidence score.

- The real-world dataset is from the Jigsaw Rate Severity of Toxic Comments on Kaggle. It has 30,108 records.
    - Citation:
    - Ian Kivlichan, Jeffrey Sorensen, Lucas Dixon, Lucy Vasserman, Meghan Graham, Tin Acosta, Walter Reade. (2021). Jigsaw Rate Severity of Toxic Comments . Kaggle. https://kaggle.com/competitions/jigsaw-toxic-severity-rating
- The intent is to share with Duc's friends and colleagues, but for those with nefarious intent, this Text Moderation model is governed by the GNU 3.0 License: https://www.gnu.org/licenses/gpl-3.0.en.html
- Author: Copyright (C), 2024 **[Duc Haba](https://linkedin.com/in/duchaba)**
---
# "AI Solution Architect" Course by ELVTR

>Welcome to the fascinating world of AI and natural language processing (NLP). This NLP model is a part of one of three hands-on application. In our journey together, we will explore the [AI Solution Architect](https://elvtr.com/course/ai-solution-architect?utm_source=instructor&utm_campaign=AISA&utm_content=linkedin) course, meticulously crafted by ELVTR in collaboration with Duc Haba. This course is intended to serve as your gateway into the dynamic and constantly evolving field of AI Solution Architect, providing you with a comprehensive understanding of its complexities and applications.

>An AI Solution Architect (AISA) is a mastermind who possesses a deep understanding of the complex technicalities of AI and knows how to creatively integrate them into real-world solutions. They bridge the gap between theoretical AI models and practical, effective applications. AISA works as a strategist to design AI systems that align with business objectives and technical requirements. They delve into algorithms, data structures, and computational theories to translate them into tangible, impactful AI solutions that have the potential to revolutionize industries.

> [Sign up for the course today](https://elvtr.com/course/ai-solution-architect?utm_source=instructor&utm_campaign=AISA&utm_content=linkedin), and I will see you in class.

- An article about this NLP Text Moderation will be coming soon.
"""
txt3 = """
## WARNING: WARNING:
---

- The following button will retrieve **real-world** offensive posts from Twitter and customer reviews from consumer companies.
- The button will display four toxic messages at a time. **Click again** for four more randomly selected postings/tweets.
- They contain **profanity, vulgarity, hate, violence, sexism, and other offensive language.**
- After you fetch the toxic messages, Click on the **"Measure 14 Toxicity" button**.
"""
#reverse_button.click(process_text, inputs=text_input, outputs=reversed_text)
#

with gradio.Blocks() as gradio_app:
  # title
 #gradioMarkdown(txt1) # any html or simple mark up
  #
  # first row, has two columns 1/3 size and 2/3 size
 with gradio.Row():    # items inside rows are columns
    # left column
  with gradio.Column(scale=1): # items under columns are row, scale is 1/3 size
      # left column has two rows, text entry, and buttons
    in1.render()
    in2.render()
    but1.render()
    out3.render()
    but1.click(monty.fetch_toxicity_level, inputs=[in1, in2], outputs=[out1,out3])

  with gradio.Column(scale=2):
    out1.render()
  #
  # second row is warning text
  with gradio.Row():
    gradio.Markdown(txt3)

  # third row is fetching toxic data
  with gradio.Row():
    with gradio.Column(scale=1):
      but2.render()
      but2.click(fetch_toxic_tweets, inputs=None, outputs=[out2, in1])
    with gradio.Column(scale=2):
      out2.render()

  # fourth row is note text
  with gradio.Row():
    gradio.Markdown(txt2)
# %%write -a app.py
# prompt: start graido_app

#gradio_app.launch(debug=True)
gradio_app.launch()