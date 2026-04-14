import os
from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

def download_game(split="train", index=0):
    games = getListGames(split=split)
    game = games[index]
    return game

def main():
    print("main started")

    repo_root = Path(__file__).resolve().parent.parent
    local_dir = repo_root / "data" / "raw"
    local_dir.mkdir(parents=True, exist_ok=True)

    password = os.getenv("SOCCERNET_PWD")
    if not password:
        raise RuntimeError("Set SOCCERNET_PWD in your terminal.")

    mydl = SoccerNetDownloader(LocalDirectory=str(local_dir))
    mydl.password = password

    game = download_game(split="train", index=0)
    print("Downloading:", game)

    mydl.downloadGame(game=game, files=["1_720p.mkv"])

    print("Download complete")
    print("Saved to:", local_dir)

if __name__ == "__main__":
    main()