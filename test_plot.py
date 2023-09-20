import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import w8HelperFunc as w8
import cv2
import numpy as np


def test_opencv_circle():
        
    
    # Reading an image in default mode
    Img = np.zeros((512, 512, 3), np.uint8)
        
    # Window name in which image is displayed
    window_name = 'Image'
        
    # Center coordinates
    center_coordinates = (220, 150)
    
    # Radius of circle
    radius = 100
        
    # Red color in BGR
    color = (255, 133, 233)
        
    # Line thickness of -1 px
    thickness = -1
        
    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    image = cv2.circle(Img, center_coordinates, radius, color, thickness)
        
    # Displaying the image
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_plot_icons():
    paths = [
        'lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg',
        # './fruit_icons/lemon_20x20.jpg'
        ]
        
    x = [0,1,2,3,4]
    y = [0,1,2,3,4]

    fig, ax = plt.subplots()
    ax.scatter(x, y) 

    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(w8.getImage('lemon_20x20.png'), (x0, y0), frameon=False)
        ax.add_artist(ab)

    plt.show()


if __name__ == '__main__':
    test_opencv_circle()
    # test_plot_icons()