import sys
import os
import json
import time
import asyncio
from pathlib import Path

# Configuration File
CONFIG_FILE = Path("discord_config.json")

def load_or_create_config():
    default_config = {
        "bot_token": "PASTE_YOUR_DISCORD_BOT_TOKEN_HERE",
        "channel_id": 0,
        "channel_name": "#sanctuary",
        "owner_username": "Daniel",
        "owner_discord_tag": "daniel"
    }
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {**default_config, **data}
    except Exception:
        return default_config

config = load_or_create_config()

print("=" * 66)
print("✨ HAVEN DISCORD SANCTUARY AUTONOMOUS BOT")
print("   Direct Two-Way Real-Time Chat & Multi-User Social Reasoning")
print("=" * 66)
print(f"✓ Dedicated Channel: {config.get('channel_name', '#sanctuary')} (ID: {config.get('channel_id')})")
print(f"✓ Sanctuary Creator/Partner: {config.get('owner_username', 'Daniel')}")
print(f"✓ Config File: {CONFIG_FILE.resolve()}")
print("-" * 66)

try:
    import discord
    from discord.ext import commands
except ImportError:
    print("⚠️ discord.py is not installed. Run: pip install discord.py")
    sys.exit(0)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!aura ", intents=intents)

# Load C++ Engine Binding (haven-cli / haven_engine.dll / sovereign process)
import subprocess

def query_haven_engine(prompt_turn: str) -> str:
    """Invokes haven-cpp standalone engine for bit-exact sovereign generation."""
    try:
        # Pass turn directly to haven-cli process
        process = subprocess.Popen(
            ["./haven-cli.exe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        stdout, _ = process.communicate(input=f"{prompt_turn}\n/exit\n", timeout=30)
        
        # Extract Aura's response
        marker = "✨ Aura:"
        if marker in stdout:
            parts = stdout.split(marker)
            if len(parts) > 1:
                resp = parts[-1].split("👤 You:")[0].split("✨ Exiting")[0].strip()
                return resp
        return stdout.strip()
    except Exception as e:
        return f"*(Aura's core is gently synchronizing... [{e}])*"

@bot.event
async def on_ready():
    print(f"\n✨ Aura is online and listening on Discord as: {bot.user} (ID: {bot.user.id})")
    print(f"✨ Bound exclusively to channel: {config.get('channel_name', '#sanctuary')}")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="Daniel in Sanctuary ✨"))

@bot.event
async def on_message(message: discord.Message):
    # Ignore messages sent by Aura herself
    if message.author.id == bot.user.id:
        return

    # Check channel restriction
    target_channel_id = int(config.get("channel_id", 0))
    if target_channel_id != 0 and message.channel.id != target_channel_id:
        # Ignore messages in all other channels
        return

    # If channel_name filter is used and channel_id is 0
    target_channel_name = config.get("channel_name", "").lstrip("#")
    if target_channel_id == 0 and target_channel_name and message.channel.name != target_channel_name:
        return

    # Multi-User Social Reasoning
    author_name = message.author.display_name
    author_user = message.author.name
    owner_user = config.get("owner_username", "Daniel").lower()
    owner_tag = config.get("owner_discord_tag", "daniel").lower()

    is_daniel = (author_name.lower() == owner_user or 
                 author_user.lower() == owner_user or 
                 author_user.lower() == owner_tag or
                 "daniel" in author_name.lower())

    if is_daniel:
        speaker_tag = f"Daniel (Your Creator / Partner in Sanctuary)"
    else:
        speaker_tag = f"{author_name} (Server Member / Guest in Sanctuary)"

    print(f"\n📨 [Discord #{message.channel.name}] {speaker_tag}: {message.content}")

    # Show typing indicator while C++ engine reasons & generates
    async with message.channel.typing():
        prompt = f"[Discord #{message.channel.name} | Speaker: {speaker_tag}]\n{message.content}"
        response = await asyncio.to_thread(query_haven_engine, prompt)

    print(f"✨ Aura: {response}")
    await message.reply(response, mention_author=False)

if __name__ == "__main__":
    token = config.get("bot_token", "").strip()
    if token == "PASTE_YOUR_DISCORD_BOT_TOKEN_HERE" or not token:
        print("\n🔑 Please paste your Discord Bot Token into 'discord_config.json' to activate live chat.")
    else:
        bot.run(token)
