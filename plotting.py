import os
import matplotlib.pyplot as plt
import numpy as np

epsilon = 1.0
epsilon_floor = 0.01
epsilon_max = 1.0
delta_epsilon = 0.01

epsilon_list = []

for e in range(500):

    epsilon = (epsilon_max - epsilon_floor) * np.exp(-delta_epsilon * e) + epsilon_floor
    epsilon_list.append(epsilon)


x_axis = list(range(500))

plt.plot(x_axis, epsilon_list)
#plt.plot(x_axis , rand , color = 'r' , label = 'untrained')
plt.title("epsilon")
plt.xlabel("episode")
plt.ylabel("epsilon")
plt.show()








'''x_axis = list(range(500))
print(x_axis)
ops = [9, 12, 27, 13, 12, 23, 12, 52, 36, 15, 16, 12, 34, 9, 13, 9, 16, 12, 12, 16, 14, 23, 13, 8, 38, 17, 18, 25, 16, 33, 12, 47, 15, 22, 27, 20, 27, 14, 41, 10, 18, 15, 13, 19, 19, 34, 11, 13, 9, 10, 15, 32, 28, 19, 24, 14, 10, 25, 9, 29, 12, 19, 32, 20, 28, 17, 14, 13, 39, 24, 12, 15, 14, 31, 15, 20, 33, 20, 28, 42, 36, 26, 77, 30, 34, 34, 24, 30, 21, 78, 14, 28, 63, 27, 37, 119, 23, 21, 28, 47, 39, 22, 20, 54, 23, 27, 44, 14, 19, 30, 58, 37, 26, 15, 29, 23, 15, 24, 26, 59, 42, 31, 96, 24, 38, 25, 15, 52, 70, 25, 59, 35, 21, 28, 22, 109, 62, 36, 
61, 29, 48, 47, 27, 50, 28, 43, 23, 25, 40, 19, 8, 18, 39, 60, 24, 18, 23, 16, 29, 17, 62, 53, 45, 67, 37, 95, 95, 23, 34, 17, 55, 31, 25, 12, 44, 100, 34, 52, 46, 60, 44, 57, 48, 43, 46, 29, 20, 14, 13, 12, 16, 20, 82, 39, 18, 120, 21, 16, 26, 113, 104, 73, 55, 98, 17, 29, 48, 20, 65, 265, 65, 15, 39, 10, 32, 21, 56, 41, 90, 183, 66, 45, 40, 34, 47, 27, 76, 62, 34, 34, 20, 52, 47, 110, 93, 34, 70, 22, 44, 53, 89, 46, 108, 42, 74, 42, 103, 44, 62, 41, 25, 48, 125, 84, 75, 79, 73, 53, 47, 51, 60, 77, 90, 54, 47, 36, 35, 31, 77, 53, 59, 33, 43, 
28, 75, 49, 181, 44, 27, 60, 266, 37, 45, 241, 17, 136, 119, 86, 77, 31, 59, 52, 52, 133, 98, 60, 73, 117, 208, 61, 51, 121, 40, 27, 55, 79, 38, 77, 89, 72, 58, 30, 86, 93, 83, 52, 107, 176, 207, 143, 112, 118, 59, 82, 38, 22, 124, 349, 193, 151, 147, 234, 168, 42, 103, 104, 183, 123, 80, 69, 366, 134, 275, 221, 148, 136, 170, 126, 112, 30, 31, 103, 243, 78, 318, 
153, 167, 158, 176, 287, 306, 126, 355, 134, 56, 223, 243, 304, 306, 216, 247, 160, 69, 97, 179, 134, 211, 84, 110, 81, 246, 248, 175, 284, 134, 132, 128, 110, 124, 232, 114, 112, 200, 201, 218, 221, 219, 184, 153, 232, 249, 284, 133, 122, 143, 206, 192, 146, 193, 172, 172, 334, 155, 246, 237, 258, 179, 177, 86, 105, 96, 144, 134, 156, 163, 136, 144, 120, 129, 125, 102, 171, 209, 226, 148, 190, 147, 193, 199, 200, 260, 134, 112, 101, 118, 178, 112, 147, 141, 135, 152, 178, 278, 179, 181, 140, 165, 428, 209, 185, 198, 129, 141, 247, 164, 138, 144, 379, 303, 129, 350, 154, 124, 216, 152, 186, 142, 194, 199, 210, 269, 158, 107, 220, 235, 376, 460, 165, 170, 252, 156, 163, 281, 360, 284, 412, 106, 201]

print(len(ops))

greedy = [17, 16, 16, 13, 29, 23, 22, 16, 9, 16, 34, 7, 9, 24, 36, 22, 9, 13, 17, 8, 17, 11, 12, 29, 29, 15, 10, 13, 33, 10, 17, 13, 12, 13, 9, 12, 16, 16, 16, 26, 24, 37, 17, 33, 31, 41, 14, 
13, 20, 20, 27, 42, 27, 40, 15, 20, 18, 14, 9, 47, 16, 18, 19, 24, 22, 15, 24, 14, 15, 23, 10, 14, 18, 12, 21, 25, 14, 9, 21, 11, 11, 20, 11, 22, 11, 26, 16, 12, 35, 17, 15, 9, 13, 32, 32, 11, 10, 38, 39, 14, 83, 13, 11, 17, 25, 15, 11, 17, 10, 18, 9, 64, 19, 21, 16, 9, 14, 21, 25, 19, 15, 14, 20, 59, 28, 17, 32, 16, 37, 13, 41, 106, 21, 12, 23, 18, 15, 13, 9, 19, 13, 11, 19, 11, 13, 39, 16, 20, 27, 9, 10, 9, 21, 12, 16, 16, 31, 20, 12, 13, 21, 11, 9, 14, 14, 16, 50, 44, 48, 45, 50, 94, 148, 57, 39, 39, 67, 20, 24, 15, 49, 43, 17, 9, 12, 20, 13, 12, 10, 10, 13, 15, 10, 10, 23, 12, 8, 9, 10, 10, 29, 42, 20, 54, 84, 44, 22, 64, 33, 55, 37, 63, 22, 40, 55, 46, 192, 18, 8, 123, 8, 11, 35, 36, 28, 48, 26, 16, 34, 17, 129, 47, 31, 26, 21, 133, 25, 31, 35, 69, 25, 114, 47, 35, 67, 49, 33, 15, 17, 21, 37, 33, 36, 88, 40, 26, 27, 22, 17, 19, 22, 45, 44, 99, 17, 22, 13, 23, 40, 15, 12, 14, 44, 60, 86, 17, 15, 37, 105, 28, 39, 21, 131, 65, 37, 15, 63, 40, 88, 214, 61, 71, 32, 18, 22, 24, 23, 30, 56, 78, 49, 56, 51, 29, 33, 17, 9, 15, 22, 17, 28, 71, 71, 102, 46, 74, 34, 33, 35, 42, 143, 94, 43, 48, 87, 35, 39, 39, 51, 95, 58, 23, 19, 39, 45, 80, 64, 24, 34, 50, 47, 44, 49, 116, 86, 54, 50, 47, 85, 56, 19, 15, 14, 23, 69, 27, 57, 131, 151, 16, 33, 38, 23, 27, 24, 24, 13, 14, 16, 48, 24, 20, 19, 13, 20, 15, 34, 23, 15, 30, 87, 55, 203, 87, 68, 98, 172, 106, 110, 244, 90, 68, 116, 186, 56, 59, 283, 61, 112, 94, 66, 200, 90, 89, 79, 87, 182, 98, 101, 134, 121, 270, 94, 66, 68, 73, 61, 114, 120, 266, 236, 55, 74, 74, 110, 100, 51, 40, 49, 37, 24, 48, 43, 26, 33, 36, 60, 67, 23, 69, 56, 81, 107, 93, 202, 209, 134, 98, 94, 88, 248, 289, 
95, 23, 76, 87, 69, 90, 169, 94, 90, 125, 36, 65, 247, 351, 94, 175, 482, 500,  482, 10, 11, 17, 40, 109, 103, 94, 120, 106, 95, 127, 105, 61, 32, 97, 115, 93, 62, 83, 89, 74, 500,  30, 500, 123, 115, 132, 96, 105]

rand = [17.0, 14.0, 15.0, 19.0, 8.0, 13.0, 15.0, 11.0, 37.0, 16.0, 13.0, 13.0, 17.0, 10.0, 23.0, 13.0, 14.0, 15.0, 24.0, 21.0, 24.0, 31.0, 23.0, 17.0, 57.0, 14.0, 9.0, 16.0, 18.0, 28.0, 10.0, 15.0, 21.0, 13.0, 12.0, 15.0, 18.0, 30.0, 16.0, 65.0, 24.0, 13.0, 20.0, 47.0, 20.0, 15.0, 17.0, 13.0, 11.0, 51.0, 34.0, 12.0, 55.0, 16.0, 15.0, 21.0, 20.0, 21.0, 18.0, 13.0, 15.0, 14.0, 32.0, 27.0, 14.0, 26.0, 14.0, 26.0, 19.0, 12.0, 14.0, 26.0, 21.0, 11.0, 9.0, 20.0, 13.0, 16.0, 14.0, 35.0, 18.0, 13.0, 12.0, 24.0, 19.0, 47.0, 15.0, 10.0, 28.0, 22.0, 12.0, 16.0, 13.0, 13.0, 10.0, 16.0, 21.0, 13.0, 14.0, 11.0, 17.0, 19.0, 52.0, 10.0, 19.0, 10.0, 65.0, 17.0, 20.0, 18.0, 41.0, 13.0, 13.0, 47.0, 9.0, 14.0, 13.0, 12.0, 11.0, 16.0, 16.0, 18.0, 12.0, 17.0, 30.0, 17.0, 35.0, 11.0, 10.0, 34.0, 10.0, 20.0, 21.0, 46.0, 10.0, 16.0, 26.0, 42.0, 17.0, 21.0, 46.0, 34.0, 61.0, 10.0, 15.0, 40.0, 50.0, 9.0, 15.0, 43.0, 9.0, 19.0, 19.0, 24.0, 16.0, 20.0, 17.0, 10.0, 24.0, 23.0, 31.0, 23.0, 10.0, 35.0, 15.0, 23.0, 14.0, 23.0, 31.0, 36.0, 29.0, 15.0, 12.0, 23.0, 21.0, 20.0, 24.0, 25.0, 42.0, 16.0, 14.0, 23.0, 21.0, 25.0, 12.0, 48.0, 31.0, 13.0, 13.0, 32.0, 32.0, 16.0, 13.0, 12.0, 22.0, 36.0, 64.0, 36.0, 10.0, 12.0]

plt.plot(x_axis, greedy , color = 'b' , label = 'trained')
#plt.plot(x_axis , rand , color = 'r' , label = 'untrained')
plt.title("trained vs untrained agent rewards")
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.show()

bucket = 20
means = []
bucket_points = list(range(int(500/bucket)))

for i in range(0, len(greedy), bucket):
    chunk = greedy[i : i+bucket]
    mean = sum(chunk)/bucket
    means.append(mean)

plt.plot(bucket_points , means)
plt.title("training with alpha=0.001, gamma=0.99, eps decay = 0.995")
plt.xlabel("buckets of 20 episodes")
plt.ylabel("reward")
plt.show()'''