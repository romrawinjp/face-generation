import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

def plot_2d(coordinate, title=""):
  plt.figure(figsize=(5,5))
  x = coordinate[:, 0]
  y = coordinate[:, 1]
  plt.ylim(max(y), min(y))
  plt.scatter(x, y)
  plt.title(title)
  plt.show()

def plot_3d(coordinate, title=""):
  x = coordinate[:, 0]
  y = coordinate[:, 1]
  z = coordinate[:, 2]
  fig = plt.figure(figsize=(5, 5))
  ax = Axes3D(fig)
  ax.scatter3D(x, z, y,  c=z, cmap='viridis', linewidth=1.0)
  ax.set_xlabel('x')
  ax.set_ylabel('z')
  ax.set_zlabel('y')
  ax.set_title(title)
  ax.view_init(190, 110)
  plt.show()

def display_point(image, face_landark, num):
    xs, ys = face_landmark[num]
    crop_img = image[ys-10:ys+10, xs-10:xs+10]
    plt.imshow(crop_img)
    plt.axis("off")
    plt.show()

# def my_function():
#   print("Hello World")

# # Defining our variable
# name = "Nicholas"

# # Defining a class
# class Student:
#   def __init__(self, name, course):
#     self.course = course
#     self.name = name

#   def get_student_details(self):
#     print("Your name is " + self.name + ".")
#     print("You are studying " + self.course)