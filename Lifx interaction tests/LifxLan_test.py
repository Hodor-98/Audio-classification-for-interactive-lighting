#!/usr/bin/env python
# coding=utf-8
import time

from lifxlan import Light, BLUE, GREEN


light = Light("d0:73:d5:01:2b:ef", "192.168.0.100")
light.set_power(1, 10, 1)
print(light.get_color())
light.set_color(BLUE)
time.sleep(5)
light.set_color(GREEN)
time.sleep(5)
light.set_power(0)