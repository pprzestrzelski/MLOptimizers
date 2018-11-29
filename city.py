import math


class City:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    @staticmethod
    def calculate_distance(city_1, city_2):
        return math.sqrt(math.pow(math.fabs(city_1.get_x() - city_2.get_x()), 2) +
                         math.pow(math.fabs(city_1.get_y() - city_2.get_y()), 2))
