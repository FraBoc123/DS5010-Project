from bs4 import BeautifulSoup
import requests
import pandas as pd
import time


class Scrapper:
    def __init__(self, url, excel_file_path):
        self.url = url
        self.excel_file_path = excel_file_path

    def _export_data(self, data):
        df = pd.DataFrame(data)
        df.to_excel(self.excel_file_path)

    def get_data(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, features="lxml")
        standings_table = soup.select('table.stats_table')[0]
        links = standings_table.find_all('a')
        links = [l.get("href") for l in links]
        links = [l for l in links if '/squads/' in l]
        t_urls = [f"https://fbref.com{l}" for l in links]
        team_urls = []
        for i in range(15):
            team_urls.append(t_urls[i])
        all_matches = []
        for team_url in team_urls:
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", "_").lower()
            data = requests.get(team_url)
            matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
            matches["Team"] = team_name
            all_matches.append(matches)
            time.sleep(5)
        match_df = pd.concat(all_matches)
        self._export_data(match_df)


if __name__ == '__main__':
    scrapper = Scrapper("https://fbref.com/en/comps/9/11160/2021-2022-Premier-League-Stats",
                        "../data/Teams.xlsx")
    scrapper.get_data()
