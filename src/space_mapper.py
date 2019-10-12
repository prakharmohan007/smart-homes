import xml.etree.ElementTree as ET
import numpy as np
import math


class SpaceMapper:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.tree = ET.parse('../data/house_map.xml')
        self.root = self.tree.getroot()

        print("[SpaceMapper] init: Get house map for subject", subject_id)
        self.house_width = 0
        self.house_height = 0
        self.space_ids = {}
        self.space_type = {}
        # self.space_type = set()
        self.ids_space = {}
        self.house_center = (0, 0)
        self.house_map = np.array([])
        print("[SpaceMapper] init: start parsing XML file.....")
        if subject_id==2:
            self.parseHouseMap2()
        elif subject_id == 1:
            self.parseHouseMap1()
        print("[SpaceMapper] init: Mapping complete!")

    def parseHouseMap1(self):
        subject_name = "subject"+str(self.subject_id)
        num_rooms = 0
        for child in self.root[self.subject_id - 1].find("rooms"):
            self.space_ids[child.tag] = num_rooms
            self.ids_space[num_rooms] = child.tag
            # self.space_type.add(child.attrib.get("type"))
            self.space_type[child.tag] = child.attrib.get("type")
            num_rooms += 1

    def parseHouseMap2(self):
        subject_name = "subject"+str(self.subject_id)
        self.house_width = int(self.root[self.subject_id-1].find("width").text)
        self.house_height = int(self.root[self.subject_id - 1].find("height").text)
        self.house_center = (int(self.root[self.subject_id - 1].find("center")[0].text),
                             int(self.root[self.subject_id - 1].find("center")[1].text))
        num_rooms = 0
        for child in self.root[self.subject_id-1].find("rooms"):
            self.space_ids[child.tag] = num_rooms
            self.ids_space[num_rooms] = child.tag
            # self.space_type.add(child.attrib.get("type"))
            self.space_type[child.tag] = child.attrib.get("type")
            num_rooms += 1

        hmap = self.root[self.subject_id-1].find("matrix").text
        hmap = list(map(int, hmap.split()))
        self.house_map = np.array(hmap)
        self.house_map = self.house_map.reshape((self.house_height, self.house_width))

    def mapCoordToSpace(self, x_coord, y_coord):
        actual_x = math.ceil(x_coord + self.house_center[0])
        actual_y = math.ceil(self.house_center[1] - y_coord)
        if actual_x < 0 or actual_x >= self.house_width or actual_y < 0 or actual_y >= self.house_height:
            space = 'N'
            ids = self.space_ids['N']
        else:
            try:
                ids = self.house_map[actual_y][actual_x]
                space = self.ids_space[ids]
            except KeyError as err:
                print("[SpaceMapper] mapCoordToSpace: Index exceptions for (", x_coord, y_coord, ")")
                print("[SpaceMapper] mapCoordToSpace: Actual_coordinates (", actual_x, actual_y, ")")
                raise

        if space == 'null':
            print(x_coord, y_coord)
        return ids, space


if __name__ == '__main__':
    sub_obj = SpaceMapper(2)
    print(sub_obj.house_width, sub_obj.house_height)
    print(sub_obj.space_ids)
    print(sub_obj.house_center)
    print(sub_obj.house_map)
    print(sub_obj.space_type)
    exit(1)
