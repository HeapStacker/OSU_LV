import numpy as np
import matplotlib.pyplot as plt

img_id = 1

def present_image(image, image_title):
    global img_id #moraš ovdje zadati da je global jer inaće python neće otkriti ovu varijablu u lokalnom scope-u
    fig.add_subplot(rows, columns, img_id)
    plt.imshow(image, cmap="gray")
    plt.title(image_title)
    plt.axis("off")
    img_id += 1

#Inicijalizacija cijelog prozora
fig = plt.figure("Image transformations", figsize=(8, 8))

rows = 2
columns = 2
image = plt.imread("lv2/road.jpg")
img = image[:,:,0].copy()  #VAŽNO!!!! jer python po novom radi move a ne copy (s id(img) MOŽEŠ PROVJERIT ADRESU)
#NAPOMENA, izgleda da python ne kopira 
# i = 4
# a = i
# print(id(a),id(i))

lighten_factor = 100
lightended_img = np.clip(img.astype(np.uint16) + lighten_factor, 0, 255) #moramo staviti da je tip uint16 jer ako je uit8, nekim pikselima nećemo nadodati lighten_factor pravilno ako su već 255

#ovako dobimo width, height od slike
#izgleda da samo za ovu sliku prikazuje s axis("on")
height, width = img.shape
quarter_img = img[:, width // 4 : width // 2]

# rotated_img = np.rot90(img, k=-1)
#ili ovako
rotated_img = img.transpose()
fig.add_subplot(rows, columns, 3)

# fliped_img = np.fliplr(img)
#ili ovako
fliped_img = img[:, ::-1] #flipa se po širini (::-1 od početka do kraja s korakom -1)

present_image(image, "Original")
present_image(lightended_img, "Lightened")
present_image(quarter_img, "Cut image")
present_image(fliped_img, "Fliped")

plt.show()
