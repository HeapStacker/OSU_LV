import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure("Image transformations", figsize=(8, 8))
plt.axis("off")
rows = 2
columns = 2
image = plt.imread("lv2/road.jpg")
img = image[:,:,0].copy()  #VAŽNO!!!! jer python po novom radi move a ne copy (s id(img) MOŽEŠ PROVJERIT ADRESU)

#NAPOMENA, izgleda da python ne kopira 
# i = 4
# a = i
# print(id(a),id(i))

lighten_factor = 100
lightended_img = np.clip(img.astype(np.uint16) + lighten_factor, 0, 255)
fig.add_subplot(rows, columns, 1)
plt.title("Lightened image")
plt.axis("off")
plt.imshow(lightended_img, cmap="gray")

height, width = img.shape
start_col = width // 4
end_col = width // 2
quarter_img = img[:, start_col:end_col]
fig.add_subplot(rows, columns, 2)
plt.title("Quarter image")
plt.axis("off")
plt.imshow(quarter_img, cmap="gray")

# rotated_img = np.rot90(img, k=-1)
#ili ovako
rotated_img = img.transpose()
fig.add_subplot(rows, columns, 3)
plt.title("Rotated image")
plt.axis("off")
plt.imshow(rotated_img, cmap="gray")


# fliped_img = np.fliplr(img)
#ili ovako
fliped_img = img[:, ::-1] #flipa se po širini (::-1 od početka do kraja s korakom -1)
fig.add_subplot(rows, columns, 4)
plt.title("Fliped image")
plt.axis("off")
plt.imshow(fliped_img, cmap="gray")


a = img.copy()
print("\n", id(a),"\n", id(img), "\n")

plt.show()
