# test-bot(bot class)
# This example requires the 'members' and 'message_content' privileged intents to function.

import discord
import random
from discord.ext import commands
from bot_logic import gen_pass
import os
import requests
from detect_objects import detect
from collections import Counter
from model import get_class
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import cast
from typing import Any # Add this import at the top
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import datetime
from datetime import datetime
description = '''An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here.'''

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
# command prefix 
bot = commands.Bot(command_prefix='+', description=description, intents=intents)
@bot.event
async def on_message(msg):
    # Ignore our own messages
    if msg.author == bot.user:
        return
    # If message is a command, let commands system handle it
    if msg.content.startswith('+'):
        await bot.process_commands(msg)
        return
    # Otherwise, try to generate AI reply
    try:
        # small guard: don't answer large attachments or empty messages
        if not msg.content or len(msg.content) > 1000:
            return
        print(f"[AI] Processing message from {msg.author}: {msg.content[:50]}...")
        reply = await generate_ai_reply(msg.content)
        print(f"[AI] Generated reply: {reply[:100] if reply else 'EMPTY'}...")
        # If reply exceeds Discord's 2000 character limit, save to a .txt and send as file
        if reply and len(reply) > 2000:
            fname = f"reply_{msg.id}.txt"
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(reply)
                await msg.channel.send("Reply too long — sending as a .txt file:", file=discord.File(fname))
            finally:
                try:
                    os.remove(fname)
                except Exception:
                    pass
        elif reply:
            await msg.channel.send(reply)
        else:
            print("[AI] Reply is empty!")
            await msg.channel.send("Hmm, I couldn't generate a proper reply right now.")
    except Exception as e:
        # Don't crash the bot for a model error; log and fallback
        print(f"[ERROR] Error generating reply: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        fallback = random.choice([
            "Hmm, I'm not sure I SERIOUSLY understood that, can you SERIOUSLY rephrase?",
            "I couldn't say a SERIOUSLY reply just now. Try again SERIOUSLY?",
        ])
        await msg.channel.send(fallback)
MYMODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
# Smaller & faster model for CPU/limited resource systems (1.5B params)
print("Loading model... this may take a while on first run")
tokenizer = AutoTokenizer.from_pretrained(MYMODEL_NAME)
# Set pad token to avoid attention mask warning
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Detect and use GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Keep the model typed as nn.Module so static analyzers know it has .generate(...)
model = AutoModelForCausalLM.from_pretrained(MYMODEL_NAME).to(device)

# Async wrapper for generation to avoid blocking the event loop

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})') # type: ignore
    print('------')

async def generate_ai_reply(user_input: str) -> str:
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    def generate_blocking(input_ids=input_ids, attention_mask=attention_mask) -> str:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=500,  # Reduced from 150 for faster generation
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Non-sampling = faster (set to True for variety)
                num_beams=1,      # No beam search = faster
                num_return_sequences=1,
            )
            # The model's output includes the prompt; slice it out
            generated = outputs[0]
            reply_tokens = generated[input_ids.shape[-1]:]
            reply = tokenizer.decode(reply_tokens, skip_special_tokens=True)
            # Clean up assistant tags if present
            reply = reply.replace("<|assistant|>", "").strip()
            return reply
    reply = await asyncio.to_thread(generate_blocking)
    # fallback if empty
    if not reply:
        return "Sorry, I couldn't think of a reply just now..."
    return reply
# adding two numbers
@bot.command()
async def add(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left + right)
# subtracting two numbers
@bot.command()
async def min(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left - right)
# multiplication two numbers
@bot.command()
async def times(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left*right)
# division two numbers
@bot.command()
async def divide(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left/right)
# exp two numbers
@bot.command()
async def exp(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left**right)


# # give local meme see python folder Data Science drive
@bot.command()
async def meme(ctx):
    # try by your self 2 min
    # img_name = random.choice(os.listdir('images'))
    with open(f'meme/meme1.jpg', 'rb') as f:
        picture = discord.File(f)
 
    await ctx.send(file=picture)

#waktu sekarang
#  
@bot.command()
async def waktu(ctx):
    await ctx.send(datetime.now())

# duck and dog API
def get_dog_image_url():
    url = 'https://random.dog/woof.json'
    res = requests.get(url)
    data = res.json()
    return data['url']
@bot.command('dog')
async def dog(ctx):
    '''Setiap kali permintaan dog (anjing) dipanggil, program memanggil fungsi get_dog_image_url'''
    image_url = get_dog_image_url()
    await ctx.send(image_url)

def get_duck_image_url():    
    url = 'https://random-d.uk/api/random'
    res = requests.get(url)
    data = res.json()
    return data['url']


@bot.command('duck')
async def duck(ctx):
    '''Setelah kita memanggil perintah bebek (duck), program akan memanggil fungsi get_duck_image_url'''
    image_url = get_duck_image_url()
    await ctx.send(image_url)

@bot.command()
async def tulis(ctx, *, my_string: str):
    with open('kalimat.txt', 'w', encoding='utf-8') as t:
        text = ""
        text += my_string
        t.write(text)

@bot.command()
async def tambahkan(ctx, *, my_string: str):
    with open('kalimat.txt', 'a', encoding='utf-8') as t:
        text = "\n"
        text += my_string
        t.write(text)

@bot.command()
async def baca(ctx):
    with open('kalimat.txt', 'r', encoding='utf-8') as t:
        document = t.read()
        await ctx.send(document)
# spamming word
@bot.command()
async def repeat(ctx, times: int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)
        
# password generator        
@bot.command()
async def pw(ctx):
    await ctx.send(f'Kata sandi yang dihasilkan: {gen_pass(10)}')
@bot.command()
async def bye(ctx):
    await ctx.send('bye \U0001f642')
@bot.command()
async def hi(ctx):
    await ctx.send('hello👋')
# coinflip
@bot.command()
async def coinflip(ctx):
    num = random.randint(1,2)
    if num == 1:
        await ctx.send('It is Head!')
    if num == 2:
        await ctx.send('It is Tail!')

# rolling dice
@bot.command()
async def dice(ctx):
    nums = random.randint(1,6)
    if nums == 1:
        await ctx.send('It is 1!')
    elif nums == 2:
        await ctx.send('It is 2!')
    elif nums == 3:
        await ctx.send('It is 3!')
    elif nums == 4:
        await ctx.send('It is 4!')
    elif nums == 5:
        await ctx.send('It is 5!')
    elif nums == 6:
        await ctx.send('It is 6!')

# welcome message
@bot.command()
async def joined(ctx, member: discord.Member):
    """Says when a member joined."""
    await ctx.send(f'{member.name} joined {discord.utils.format_dt(member.joined_at)}') # type: ignore

#show local drive    
@bot.command()
async def local_drive(ctx):
    try:
      folder_path = "./files"  # Replace with the actual folder path
      files = os.listdir(folder_path)
      file_list = "\n".join(files)
      await ctx.send(f"Files in the files folder:\n{file_list}")
    except FileNotFoundError:
      await ctx.send("Folder not found.") 
#show local file
@bot.command()
async def showfile(ctx, filename):
  """Sends a file as an attachment."""
  folder_path = "./files/"
  file_path = os.path.join(folder_path, filename)

  try:
    await ctx.send(file=discord.File(file_path))
  except FileNotFoundError:
    await ctx.send(f"File '{filename}' not found.")
# upload file to local computer
@bot.command()
async def simpan(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            # file_url = attachment.url  IF URL
            await attachment.save(f"./files/{file_name}")
            await ctx.send(f"Menyimpan {file_name}")
    else:
       await ctx.send("Anda lupa mengunggah :(")
#Computer Vision Classification Pipit/Merpati
@bot.command()
async def klasifikasi(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            #file_url = attachment.url IF URL
            await attachment.save(f"./CV/{file_name}")
            await ctx.send(get_class(model_path="keras_model.h5", labels_path="labels.txt", image_path=f"./CV/{file_name}"))
    else:
        await ctx.send("Anda lupa mengunggah gambar :(")
#Computer Vision Deteksi objek
@bot.command()
async def deteksi(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            await attachment.save(f"./CV/{file_name}")
            # call detect ONCE and reuse its result
            results = detect(input_image=f"./CV/{file_name}", output_image=f"./CV/{file_name}", model_path="yolov3.pt")
            # If detect returns a string (message/path), send it
            if isinstance(results, str):
                await ctx.send(results)
            # If detect returns a list of detections, count them
            if isinstance(results, list):
                counts = Counter(d['name'] for d in results)
                msg = '\n'.join(f"{k}: {v}" for k, v in counts.items())
                with open(f'CV/{file_name}', 'rb') as f:
                    picture = discord.File(f)
                await ctx.send(file=picture)
                await ctx.send(f"Object counts:\n{msg}")
    else:
        await ctx.send("Anda lupa mengunggah gambar :(")

bot.run('YOUR_BOT_TOKEN_HERE')



