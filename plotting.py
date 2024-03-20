import os
import matplotlib.pyplot as plt
import numpy as np

x_axis = np.arange(200)

greedy = [112.0, 94.0, 123.0, 131.0, 117.0, 175.0, 121.0, 55.0, 154.0, 90.0, 150.0, 118.0, 119.0, 135.0, 180.0, 92.0, 124.0, 84.0, 117.0, 154.0, 146.0, 116.0, 101.0, 149.0, 102.0, 144.0, 132.0, 137.0, 106.0, 158.0, 260.0, 146.0, 127.0, 107.0, 124.0, 104.0, 137.0, 96.0, 189.0, 103.0, 111.0, 120.0, 122.0, 109.0, 96.0, 119.0, 145.0, 125.0, 138.0, 126.0, 118.0, 128.0, 124.0, 77.0, 126.0, 137.0, 108.0, 114.0, 125.0, 128.0, 118.0, 124.0, 126.0, 232.0, 128.0, 124.0, 156.0, 135.0, 115.0, 92.0, 135.0, 137.0, 127.0, 123.0, 156.0, 77.0, 125.0, 121.0, 88.0, 196.0, 102.0, 109.0, 191.0, 127.0, 124.0, 105.0, 109.0, 115.0, 130.0, 121.0, 117.0, 126.0, 127.0, 116.0, 117.0, 213.0, 198.0, 109.0, 105.0, 110.0, 93.0, 117.0, 127.0, 143.0, 49.0, 113.0, 147.0, 123.0, 112.0, 106.0, 106.0, 92.0, 47.0, 163.0, 105.0, 96.0, 124.0, 76.0, 113.0, 62.0, 128.0, 125.0, 101.0, 172.0, 113.0, 129.0, 148.0, 145.0, 124.0, 142.0, 115.0, 133.0, 126.0, 112.0, 130.0, 94.0, 113.0, 138.0, 126.0, 113.0, 128.0, 134.0, 148.0, 149.0, 69.0, 125.0, 129.0, 124.0, 78.0, 48.0, 161.0, 133.0, 127.0, 134.0, 125.0, 107.0, 124.0, 180.0, 149.0, 126.0, 152.0, 160.0, 101.0, 96.0, 123.0, 114.0, 104.0, 158.0, 113.0, 115.0, 146.0, 74.0, 144.0, 129.0, 117.0, 130.0, 112.0, 112.0, 131.0, 106.0, 131.0, 102.0, 97.0, 132.0, 108.0, 127.0, 129.0, 110.0, 105.0, 134.0, 118.0, 124.0, 100.0, 89.0, 108.0, 114.0, 138.0, 232.0, 133.0, 134.0]

rand = [17.0, 14.0, 15.0, 19.0, 8.0, 13.0, 15.0, 11.0, 37.0, 16.0, 13.0, 13.0, 17.0, 10.0, 23.0, 13.0, 14.0, 15.0, 24.0, 21.0, 24.0, 31.0, 23.0, 17.0, 57.0, 14.0, 9.0, 16.0, 18.0, 28.0, 10.0, 15.0, 21.0, 13.0, 12.0, 15.0, 18.0, 30.0, 16.0, 65.0, 24.0, 13.0, 20.0, 47.0, 20.0, 15.0, 17.0, 13.0, 11.0, 51.0, 34.0, 12.0, 55.0, 16.0, 15.0, 21.0, 20.0, 21.0, 18.0, 13.0, 15.0, 14.0, 32.0, 27.0, 14.0, 26.0, 14.0, 26.0, 19.0, 12.0, 14.0, 26.0, 21.0, 11.0, 9.0, 20.0, 13.0, 16.0, 14.0, 35.0, 18.0, 13.0, 12.0, 24.0, 19.0, 47.0, 15.0, 10.0, 28.0, 22.0, 12.0, 16.0, 13.0, 13.0, 10.0, 16.0, 21.0, 13.0, 14.0, 11.0, 17.0, 19.0, 52.0, 10.0, 19.0, 10.0, 65.0, 17.0, 20.0, 18.0, 41.0, 13.0, 13.0, 47.0, 9.0, 14.0, 13.0, 12.0, 11.0, 16.0, 16.0, 18.0, 12.0, 17.0, 30.0, 17.0, 35.0, 11.0, 10.0, 34.0, 10.0, 20.0, 21.0, 46.0, 10.0, 16.0, 26.0, 42.0, 17.0, 21.0, 46.0, 34.0, 61.0, 10.0, 15.0, 40.0, 50.0, 9.0, 15.0, 43.0, 9.0, 19.0, 19.0, 24.0, 16.0, 20.0, 17.0, 10.0, 24.0, 23.0, 31.0, 23.0, 10.0, 35.0, 15.0, 23.0, 14.0, 23.0, 31.0, 36.0, 29.0, 15.0, 12.0, 23.0, 21.0, 20.0, 24.0, 25.0, 42.0, 16.0, 14.0, 23.0, 21.0, 25.0, 12.0, 48.0, 31.0, 13.0, 13.0, 32.0, 32.0, 16.0, 13.0, 12.0, 22.0, 36.0, 64.0, 36.0, 10.0, 12.0]

plt.plot(x_axis, greedy , color = 'b' , label = 'trained')
plt.plot(x_axis , rand , color = 'r' , label = 'untrained')
plt.title("trained vs untrained agent rewards")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()