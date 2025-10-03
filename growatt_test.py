import re

from growatt import Growatt

from config import Config

"""
Add GrowattCn class to support Chinese server.
基本来自这个网站：https://server-cn.growatt.com/
"""
class GrowattCn(Growatt):
    BASE_URL = "https://server-cn.growatt.com"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 覆盖父类的实例属性，强制使用子类的类属性
        self.BASE_URL = type(self).BASE_URL  # 或者 self.__class__.BASE_URL

    def getStorageStatusData(self, plantId: str, storageSn: str):
        """
        Retrieves the storage status data for a specific plant.

        Args:
            plantId (str): The ID of the plant.

        Returns:
            dict: A dictionary containing storage status data.
        """
        res = self.session.post(f"{self.BASE_URL}/panel/storage/getStorageStatusData?plantId={plantId}",
                                data={"storageSn": storageSn})
        res.raise_for_status()

        try:
            json_res = res.json()['obj']
            return json_res
        except re.exceptions.JSONDecodeError:
            raise ValueError("Invalid response received. Please ensure you are logged in.")

    def getDevicesByPlant(self, plantId: str):
        """
        Retrieves the storage status data for a specific plant.

        Args:
            plantId (str): The ID of the plant.

        Returns:
            dict: A dictionary containing storage status data.
        """
        res = self.session.post(f"{self.BASE_URL}/panel/getDevicesByPlant?plantId={plantId}")
        res.raise_for_status()

        try:
            json_res = res.json()['obj']
            return json_res
        except re.exceptions.JSONDecodeError:
            raise ValueError("Invalid response received. Please ensure you are logged in.")


if __name__ == '__main__':

    api = GrowattCn()

    api.login(Config.GROWATT_CN_USERNAME, Config.GROWATT_CN_PASSWORD)

    plantId = api.get_plants()[0]["id"]
    print(f"Plant ID: {plantId}")

    plant = api.get_plant(plantId)
    print(plant)

    devices = api.getDevicesByPlant(plantId)
    print(devices)

    # mixSn = api.get_mix_ids(plantId)[0][1]
    # mixSn = api.get_mix_ids(plantId)
    # print(mixSn)
    #
    # mixTotal = api.get_mix_total(plantId, mixSn)
    # print(mixTotal)
    #
    # mixStatus = api.get_mix_status(plantId, mixSn)
    # print(mixStatus)

    data = api.getStorageStatusData(plantId, devices["storage"][0][0])
    print(data)
