#Imports
import tensorflow as tf
import pygame
from pygame import gfxdraw
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Import the fashion mnist library
mnist = tf.keras.datasets.mnist

#Create the training and test split.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Normalize the training and test data so that each value representing a pixel is between 0 and 1.
train_images = train_images/255.0
test_images = test_images/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=5)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#Predict the number drawn in a given image
def return_number(img):
    img = img/255.0
    img = (np.expand_dims(img, 0))
    probability_list = tf.nn.softmax(probability_model.predict(img)).numpy().flatten().tolist()
    return probability_list.index(max(probability_list))

#Pygame inits
pygame.init()
pygame.font.init()

#Create times new roman font object
times = pygame.font.SysFont("Times New Roman", 15)

#Setup the game screen
screen = pygame.display.set_mode((800,800), pygame.SRCALPHA, 32)
canvas = pygame.Surface((400,400), pygame.SRCALPHA, 32)
canvas.fill((0,0,0))

run_game = True

#Generate initial text to blit to the screen
expected_num = times.render("The number you're writing is probably: N/A", False, (0,0,0))
explanatory_text = times.render("Press space to clear the canvas and left mouse to draw", False, (0,0,0))

#Main game loop
while run_game:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run_game = False

        #Get the position of the mouse
        pos = list(pygame.mouse.get_pos())
        screen.fill((255,255,255))
        
        #Draw the canvas on the screen
        screen.blit(canvas, (200,200))

        #If the mouse is down, draw where the mouse is on the canvas
        if pygame.mouse.get_pressed()[0]:
            pygame.draw.circle(canvas, (255,255,255,0), ((pos[0])-200,(pos[1])-200), 20)
        
        #If the mouse button goes up, predict the current number written on the canvas
        #cv2.resize(np.array(cv2.flip(cv2.rotate(pygame.surfarray.array2d(canvas), cv2.ROTATE_90_CLOCKWISE),1), dtype='uint8'), dsize=(28,28), interpolation=cv2.INTER_LANCZOS4)
            #pygame.surfarray.array2d(canvas): Get the canvas as a numpy array
            #cv2.rotate(pygame.surfarray.array2d(canvas), cv2.ROTATE_90_CLOCKWISE): Rotate the canvas numpy array 90 degrees counter clockwise
            #cv2.flip(cv2.rotate(pygame.surfarray.array2d(canvas), cv2.ROTATE_90_CLOCKWISE),1): Flip the canvas horizontally
            #np.array(cv2.flip(cv2.rotate(pygame.surfarray.array2d(canvas), cv2.ROTATE_90_CLOCKWISE),1), dtype='uint8'): Change the dtype of the current numpy array to uint8
            #cv2.resize(np.array(cv2.flip(cv2.rotate(pygame.surfarray.array2d(canvas), cv2.ROTATE_90_CLOCKWISE),1), dtype='uint8'), dsize=(28,28), interpolation=cv2.INTER_LANCZOS4): Resize the canvas image down to a size of 28 by 28
        if event.type == pygame.MOUSEBUTTONUP:
            expected_num = times.render("The number you're writing is probably: " + str(return_number(cv2.resize(np.array(cv2.flip(cv2.rotate(pygame.surfarray.array2d(canvas), cv2.ROTATE_90_CLOCKWISE),1), dtype='uint8'), dsize=(28,28), interpolation=cv2.INTER_LANCZOS4))), False, (0,0,0))

        #Clear the canvas if space is spressed
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                canvas.fill((0,0,0))

        #Blit text to the screen
        screen.blit(expected_num, (200, 600))
        screen.blit(explanatory_text, (200, 650))
        
        #Update the screen
        pygame.display.update()