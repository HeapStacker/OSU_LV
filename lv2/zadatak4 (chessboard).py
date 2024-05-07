import numpy as np
import matplotlib.pyplot as plt

black_square = np.zeros((50, 50))
white_square = np.ones((50, 50)) * 255

# top_row = np.hstack((black_square, white_square))
# middle_row = np.hstack((white_square, black_square))
# bottom_row = np.hstack((black_square, white_square))
# final_image = np.vstack((top_row, middle_row, bottom_row))

def createSquareField(width = 2, height = 2):
    row_stacks = []
    for i in range(0, height):
        column_stacks = []
        for j in range(0, width):
            if (i + j) % 2 == 0:
                column_stacks.append(black_square)
            else:
                column_stacks.append(white_square)
        row_stacks.append(np.hstack(column_stacks))
    field = np.vstack(row_stacks)
    return field



plt.figure("White/black boxes")
plt.title("Kitchen floor")
plt.imshow(createSquareField(8, 8)) #možeš staviti argument cmap="gray" da bi slika ispala crno-bijela
plt.axis("off")
plt.show()