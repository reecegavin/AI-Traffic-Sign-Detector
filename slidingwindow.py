# CE4041 Project 3
# HOG SVM Classifier
# Reece Gavin ID: 17197589



# Import all necessary libraries
from PIL import Image
import matplotlib as plt
from matplotlib import image as im
from matplotlib import pyplot
from skimage.feature import hog
import numpy as np
import pickle
import sys

# The terminal is used to feed in a used defined image from the testimages folder
user_image = sys.argv[1]
image = np.array(Image.open(user_image))

# Pickle is used to open the previously written linsvmhog.pkl file and the data is stored in variable hogsvm
hogsvm = pickle.load(open("linsvmhog.pkl", "rb"))

# The image width and height is stored in appropriate variable and the window step for the sliding window
# is defined as 2
image_height = image.shape[0]
image_width = image.shape[1]
window_step = 2


# A function called sliding_window is created which accepts an image, window size, svm_index, pixels and color
def sliding_window(image, window_size, svm_index, pixels, color):
    # A number of empty arrays are created for use in the for loops below
    return_values = []
    j = []
    xx = []
    yy = []
    x_new = []
    y_new = []
    j_new = []


    # A for loop is created for when y is in the range of 0 to (image height - window size), in steps of window step
    # This for loop is used to return 0 if no detections are found for a particular window size
    for y in range(0, image_height - window_size, window_step):

        # A nested for loop is created with the same conditions as above
        for x in range(0, image_width - window_size, window_step):

            # Window is to image[y:y + window_size, x:x + window_size]
            window = image[y:y + window_size, x:x + window_size]

            # A new hog, hog_v is created with the below parameters
            hog_v = (hog(window, orientations=9, pixels_per_cell=(pixels, pixels),
                         cells_per_block=(2, 2), visualize=False, multichannel=True))

            # hog_v is reshaped to be 1 and -1
            hog_v = hog_v.reshape(1, -1)

            # The variable hog_prediction is assigned the prediction of hogsvm[svm_index]
            hog_prediction = hogsvm[svm_index].predict(hog_v)

            # The variable hog_decision is assigned the decision function of hogsvm[svm_index]
            hog_decision = hogsvm[svm_index].decision_function(hog_v)

            # If the prediction comes back = to 1 ,the  hog_decision is added to the j array
            if hog_prediction == 1:
                j.append(hog_decision)

                # The x and x + window_size are added to the xx list
                xx.append([x, x + window_size])
                # The y and y + window_size are added to the yy list
                yy.append([y, y + window_size])

    # If no detection occurs for a particular window size, return zero
    if len(j) == 0:
        return 0

    # Variables p and i initialised to 0
    p = 0
    i = 0

    while p < 2:
        # If p does not equal 0 the below variables are assigned or initialised to different values
        if p != 0:
            xx = x_new
            yy = y_new
            x_new = []
            y_new = []
            j = j_new
            j_new = []

        # j is made into a numpy array
        j = np.array(j)

        # The variable maximum is used to store the max value in j
        maximum = np.max(j)
        # max_index is assigned the indices of the maximum values along j
        max_index = j.argmax()

        # xx and yy are made into numpy arrays
        xx = np.array(xx)
        yy = np.array(yy)

        # x_detect and y_detect are assigned the values of xx[max_index, 0] and yy[max_index, 0]
        x_detect = xx[max_index, 0]
        y_detect = yy[max_index, 0]

        # return_values is appended to included the variables shown
        return_values.append([x_detect, y_detect, maximum, window_size])

        # The bounding box is drawn on the image
        ax1.add_patch(
            plt.patches.Rectangle((x_detect, y_detect), window_size, window_size, edgecolor=color, facecolor='none'))

        while i < len(j):
            # The below is used to store the boolean result of comparisons
            res1 = x_detect <= xx[i, 1]
            res2 = x_detect >= xx[i, 0]
            res3 = y_detect <= yy[i, 1]
            res4 = y_detect >= yy[i, 0]
            res5 = (y_detect + window_size) <= yy[i, 1]
            res6 = (y_detect + window_size) >= yy[i, 0]
            res7 = (x_detect + window_size) <= xx[i, 1]
            res8 = (x_detect + window_size) >= xx[i, 0]

            if ((res1 and res2) or (res7 and res8)) and ((res5 and res6) or (res3 and res4)):
                i += 1
            else:
                # Variables below have the values added to them and i is incremented by 1
                x_new.append([xx[i, 0], xx[i, 1]])
                y_new.append([yy[i, 0], yy[i, 1]])
                j_new.append(j[i])
                i += 1

        if len(x_new) != 0:
            p = 1
        else:
            p = 2

    return return_values


# Function best_fit_window is used to draw the best fit bounding box around the detections
# It takes different window sizes as parameters
def best_fit_window(wind_64, wind_48, wind_40, wind_32):
    # An empty array is created called wind
    wind = []

    # if statements used to store if a detection is made for a particular window size in the wind array
    if wind_64 != 0:
        wind = wind_64
    if wind_48 != 0:
        wind = wind + wind_48
    if wind_40 != 0:
        wind = wind + wind_40
    if wind_32 != 0:
        wind = wind + wind_32

    # Empty arrays are created
    k = []
    xx = []
    yy = []
    x_new = []
    y_new = []
    k_new = []
    return_values = []
    l = 0
    #i = 0

   #wind = np.array(wind)

    #A number of different operations are carried out the the arrays to reshape the data
    wind=np.array(wind)
    xx.append(wind[:, 0])
    xx.append(wind[:, 0] + wind[0][3])
    xx.append(wind[:, 3])
    xx = np.array(xx)
    xx = np.transpose(xx)
    xx = xx.tolist()
    yy.append(wind[:, 1])
    yy.append(wind[:, 1] + wind[0][3])
    yy = np.array(yy)
    yy = np.transpose(yy)
    yy = yy.tolist()
    k.append(wind[:, 2])
    k = np.array(k)
    k = np.transpose(k)
    k = k.tolist()

    while l < 2:

        i = 0
        # If l does not equal 0 the below variables are assigned or initialised to different values
        if l != 0:

            xx = x_new
            yy = y_new
            x_new = []
            y_new = []
            k = k_new
            k_new = []



        # k is made into a numpy array
        j = np.array(k)

        # The variable maximum is used to store the max value in j
        maximum = np.max(j)
        # max_index is assigned the indices of the maximum values along j
        max_index = j.argmax()

        # xx and yy are made into numpy arrays
        xx = np.array(xx)
        yy = np.array(yy)

        # x_detect and y_detect are assigned the values of xx[max_index, 0] and yy[max_index, 0]
        x_detect = xx[max_index, 0]
        y_detect = yy[max_index, 0]

        # if maximum is greater than 0.2 the return_values array is appended with the below variable
        if maximum > 0.2:
            return_values.append([x_detect, y_detect, maximum, xx[max_index, 2]])

        while i < len(k):
            # The below is used to store the boolean result of comparisons
            res1 = x_detect <= xx[i, 1]
            res2 = x_detect >= xx[i, 0]
            res3 = y_detect <= yy[i, 1]
            res4 = y_detect >= yy[i, 0]
            res5 = (y_detect + xx[i, 2]) <= yy[i, 1]
            res6 = (y_detect + xx[i, 2]) >= yy[i, 0]
            res7 = (x_detect + xx[i, 2]) <= xx[i, 1]
            res8 = (x_detect + xx[i, 2]) >= xx[i, 0]

            if ((res1 and res2) or (res7 and res8)) and ((res5 and res6) or (res3 and res4)):
                i += 1
            else:
                # Variables below have the values added to them and i is incremented by 1
                x_new.append([xx[i, 0], xx[i, 1], xx[i, 2]])
                y_new.append([yy[i, 0], yy[i, 1]])
                k_new.append(k[i])

                i += 1

        if len(x_new) != 0:
            l = 1
        else:
            l = 2

    return return_values

# The user image is plotted showing all detections
img = im.imread(user_image)
figure1, ax1 = pyplot.subplots(1)
figure1.suptitle('All detections')
ax1.imshow(img)


# A hog vector with window size 64 is created, svm index 0, 8 pixels per cell and a blue bounding box drawn around it
hog_vector64 = sliding_window(image, 64, 0, 8, 'b')
print("This is hog_vector64", hog_vector64)

# A hog vector with window size 48 is created, svm index 1, 6 pixels per cell and a red bounding box drawn around it
hog_vector48 = sliding_window(image, 48, 1, 6, 'r')
print("This is hog_vector48", hog_vector48)

# A hog vector with window size 40 is created, svm index 2, 4 pixels per cell and a yellow bounding box drawn around it
hog_vector40 = sliding_window(image, 40, 2, 4, 'y')
print("This is hog_vector40", hog_vector40)

# A hog vector with window size 32 is created, svm index 3, 4 pixels per cell and a black bounding box drawn around it
hog_vector32 = sliding_window(image, 32, 3, 4, 'k')
print("This is hog_vector32", hog_vector32)

# The best_fit_window is called and the 4 different vectors passed into it to get the best detections
final = best_fit_window(hog_vector64, hog_vector48, hog_vector40, hog_vector32)

# The img variable reads in the user defined image and plots on on a figure
img = im.imread(user_image)
figure2, ax = pyplot.subplots(1)
figure2.suptitle('Best detections')
ax.imshow(img)

# A variable p_numb is initialised. It will be used to store the patch number for bounding boxes
p_numb = 0

# The following while loop is used to print out the x and y values of a detection, its activation level and the window size
while p_numb < len(final):
    print('x,y value for detection \t Activation Level \t Window Size')
    print(final[p_numb][0], final[p_numb][1], '\t', final[p_numb][2], '\t',
          final[p_numb][3])
    '''print('Activation Level')
    print(final_result[p_numb][2])
    print('Window Size')
    print(final_result[p_numb][3])'''

    # The bounding boxes are drawn on the best fit detection with a yellow bounding box
    ax.add_patch(plt.patches.Rectangle((final[p_numb][0], final[p_numb][1]),
                                       final[p_numb][3], final[p_numb][3], edgecolor='y',
                                        facecolor='none'))


    # p_numb is incremented
    p_numb += 1

# The plots are shown
pyplot.show()
