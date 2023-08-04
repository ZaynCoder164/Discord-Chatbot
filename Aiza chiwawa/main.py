import discord
from discord import Activity, ActivityType
import asyncio
import os
from itertools import cycle
from discord.ext import commands
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
from datetime import datetime
import json
import pickle
import PyLyrics 
import wikipediaapi
from dotenv import load_dotenv
from utilities.config_loader import config, load_current_language

stemmer = LancasterStemmer()
load_dotenv()

# Set up the Discord bot
intents = discord.Intents.all()
TOKEN = os.getenv('DISCORD_TOKEN')  # Loads Discord bot token from env

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = nltk.word_tokenize(doc)
        wrds = [stemmer.stem(w.lower()) for w in wrds]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model_path = "model.tflearn"

# Replace the model loading code with the following
# try:
#     model.load(model_path)
# except tf.errors.NotFoundError:
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save(model_path)

bot = commands.Bot(command_prefix='!', intents=intents)
# Discord config
allow_dm = config['ALLOW_DM']
trigger_words = config['TRIGGER']
smart_mention = config['SMART_MENTION']
presences = config["PRESENCES"]

# Instructions Loader
current_language = load_current_language()

@bot.event
async def on_ready():
    await bot.tree.sync()
    presences_cycle = cycle(presences + [current_language['help_footer']])
    print(f"{bot.user} aka {bot.user.name} has connected to Discord!")
    invite_link = discord.utils.oauth_url(
        bot.user.id,
        permissions=discord.Permissions(),
        scopes=("bot", "applications.commands")
    )
    print(f"Invite link: {invite_link}")
    print()
    print()
    while True:
        presence = next(presences_cycle)
        presence_with_count = presence.replace("{guild_count}", str(len(bot.guilds)))
        delay = config['PRESENCES_CHANGE_DELAY']
        
        # Set combined activity with both "Playing" and "Listening"
        combined_activity = Activity(type=ActivityType.playing, name=presence_with_count)
        combined_activity_listening = Activity(type=ActivityType.listening, name='My Ordinary Life - The Living Tombstone')

        await bot.change_presence(status=discord.Status.online, activity=combined_activity)
        await asyncio.sleep(delay)
        await bot.change_presence(status=discord.Status.online, activity=combined_activity_listening)
        await asyncio.sleep(delay)

target_channel_id = 1123179892612276284

# Create a Wikipedia API object with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="Aiza/1.0"
)

# Import the necessary library
from googlesearch import search
from bs4 import BeautifulSoup

def get_top_trending_music():
    query = "top trending music"  # The search query
    num_results = 10  # Number of links to fetch

    top_music_links = []
    for link in search(query, num_results=num_results):
        top_music_links.append(link)

    return top_music_links

def parse_song_and_artist(user_input):
    # Assuming the input_text is in the format "song name - artist name"
    # You can change the delimiter and parsing logic as per your needs
    delimiter = "-"
    parts = user_input.split(delimiter, 1)  # Split at the first occurrence of the delimiter

    if len(parts) == 2:
        song_name = parts[0].strip()
        artist_name = parts[1].strip()
    else:
        # If the input format is incorrect, you can set some default values or return None
        song_name = None
        artist_name = None

    return song_name, artist_name

from PyLyrics import *
def generate_lyrics(song_name, artist_name):
    try:
        lyrics = PyLyrics.get_lyrics(artist_name, song_name)
        return lyrics
    except ValueError:
        return "Lyrics not found for this song."

    
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_input = message.content.lower()

   # Check if the message is from a DM
    if isinstance(message.channel, discord.DMChannel):
        # Check if the message starts with the command "tell"
        if user_input.startswith('$tell'):
            search_query = user_input[len('$tell'):].strip()  # Extract the query after the "tell" command
            page = wiki_wiki.page(search_query)
            if page.exists():
                response = page.summary[:2000]  # Truncate the summary to Discord's character limit
            elif user_input.notstartswith('$tell'):
                response ="Sorry, Try adding '$' as a prefix for tell"
            else:
                response = "Sorry, I couldn't find any relevant information."

        elif message.content.startswith('$lyrics'):
            user_input = message.content[len('$lyrics'):].strip()
            song_name, artist_name = parse_song_and_artist(user_input)
            response = generate_lyrics(song_name, artist_name)

            if response:
                await message.channel.send(response)
            else:
                await message.channel.send("Lyrics not found for this song.")
        
        elif any(word in user_input for word in ['time', 'current time', 'tell me the time']):
            current_time = datetime.now().strftime("%I:%M:%S %p")
            response = random.choice([
                f"The current time is {current_time}.",
                f"It is currently {current_time}.",
                f"The time is {current_time}."
            ])
                
        else:
            # Process the user input and generate a response using the neural network model
            results = model.predict([bag_of_words(user_input, words)])[0]
            results_index = np.argmax(results)
            tag = labels[results_index]

            if results[results_index] > 0.6:  # Set the threshold to 0.6 (60% confidence)
                for tg in data["intents"]:
                    if tg["tag"] == tag:
                        responses = tg["responses"]
                        if responses:  # Check if responses list is not empty
                            response = random.choice(responses)
                        else:
                            response = "Sorry, I didn't get that, try again."
                        break
                    else:
                        response = "Sorry, I didn't get that, try again."
            else:
                response = "Sorry, I didn't get that, try again."


    else:
        # Check if the message starts with the command "$" in public channels
        if message.content.startswith('$'):
            # Extract the query after the "$" command
            search_query = user_input[len('$'):].strip()
            print("Search Query:", search_query)
            await message.add_reaction('🤔')  # Add thinking reaction

            if "tell" in search_query:
                search_query = search_query.replace("tell", "").strip()  # Remove "tell" from the query
                page = wiki_wiki.page(search_query)
                await message.remove_reaction('🤔', bot.user)  # Remove thinking reaction after processing
                if page.exists():
                    response = page.summary[:2000]  # Truncate the summary to Discord's character limit
                elif "tell" not in search_query:
                    response ="Sorry, type tell before the query"
                else:
                    response = "Sorry, I couldn't find any relevant information."
                    
            elif any(word in user_input for word in ['time', 'current time', 'tell me the time']):
                current_time = datetime.now().strftime("%I:%M:%S %p")
                response = random.choice([
                f"The current time is {current_time}.",
                f"It is currently {current_time}.",
                f"The time is {current_time}."
                 ])

            else:
                await message.remove_reaction('🤔', bot.user)  # Remove thinking reaction if the query is not a Wikipedia search

                # Process the user input and generate a response using the neural network model
                results = model.predict([bag_of_words(user_input, words)])[0]
                results_index = np.argmax(results)
                tag = labels[results_index]

                if results[results_index] > 0.6:  # Set the threshold to 0.6 (60% confidence)
                    for tg in data["intents"]:
                        if tg["tag"] == tag:
                            responses = tg["responses"]
                            if responses:  # Check if responses list is not empty
                                response = random.choice(responses)
                            else:
                                response = "Sorry, I didn't get that, try again."
                            break
                    else:
                        response = "Sorry, I didn't get that, try again."
                else:
                    response = "Sorry, I didn't get that, try again."

        elif message.content.startswith('$lyrics'):
            user_input = message.content[len('$lyrics'):].strip()
            song_name, artist_name = parse_song_and_artist(user_input)
            response = generate_lyrics(song_name, artist_name)

            if response:
                await message.channel.send(response)
            else:
                await message.channel.send("Lyrics not found for this song.")

        else:
            return

    await message.channel.send(response)  # Move this line outside the if-else blocks
       
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


bot.run(TOKEN)
