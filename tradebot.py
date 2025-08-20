import json
import logging
import aiohttp
from datetime import datetime, timedelta

import botpy
from botpy.message import Message, GroupMessage

_log = logging.getLogger()

class MyClient(botpy.Client):
    async def on_ready(self):
        _log.info(f"robot 「{self.robot.name}」 on_ready!")
    async def on_at_message_create(self, message: Message):
        await message.reply(content=f"机器人{self.robot.name}收到你的@消息了: {message.content}")

    async def on_group_at_message_create(self, message: GroupMessage):
        # Determine whether to request today's or yesterday's data based on message content
        if "前一天" in message.content or "昨天" in message.content:
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            target_date = datetime.now().strftime('%Y-%m-%d')

        
        # Request data from the API
        # url = f"http://localhost:8080/api/select/date/{target_date}"
        url = f"http://host.docker.internal:9966/api/select/date/{target_date}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Summarize the data
                    summary = self.summarize_data(data, target_date)
                    await message.reply(content=summary)
                else:
                    await message.reply(content="获取数据失败")

    def summarize_data(self, data, date):
        # Summarize the data into a readable format
        if not data:
            return f"{date}日期暂无数据"
        summary_lines = []
        summary_lines.append(f"日期: {date}")
        for item in data:
            summary_lines.append(f"策略: {item['strategy']}")
            stocks = json.loads(item['stockSelectionResult'])
            summary_lines.append(f"选股结果: {', '.join(stocks)}")
        return '\n'.join(summary_lines)


if __name__ == '__main__':
    with open('bot_config.json', 'r') as f:
        config = json.load(f)
    client = MyClient(intents=botpy.Intents(public_messages=True))
    client.run(appid=config['appId'], secret=config['appSecret'])
