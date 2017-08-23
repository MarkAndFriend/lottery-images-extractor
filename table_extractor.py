
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter

def non_maximun_suppression(array,window_scale = 25):
    """
        do non maximun suppression
        
        Parameters
        ----------
            array : like array
                array with have features for do a non-maximun-suppression
                
            window_scale : int
                scale of window which will do a non-maximun-suppression
                default : 25
            
        Return
        ------
            like array
                result from non-maximun-suppression of this array
    """
    (col_size , row_size) = array.shape
    # for col in range(col_size):
    #     for row in range(row_size):
    #         
        
    return array
    
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def non_max_suppression(input, window_size = 25):
    # input: B x W x H x C
    pooled = tf.nn.max_pool(input, ksize=[window_size, window_size], strides=[1,1,1,1], padding='SAME')
    output = tf.where(tf.equal(input, pooled), input, tf.zeros_like(input))

    # NOTE: if input has negative values, the suppressed values can be higher than original
    return output # output: B X W X H x C
    
    
class TableImage(object):
    """
        table image object
    """
    
    def __init__(self,image):
        """
            Construct Table image from image
            
            Parameters
            ----------
                image : Image
                
        """
        self.edges = cv2.Canny(img,100,200)
        
        self.edges = TableImage.detectCrossEdges(self.edges)
        
        return
        
    @staticmethod
    def detectCrossEdges(edges):
        """
            detect cross edges
            
            Return
            ------
                cross edges : like array
                    new edge image with show only cross part    
        """
        # kernel = np.array([  
        #             [0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[.25,.25,.25,.25,.25,.5,.75,1,1,1,1,.75,.5,.25,.25,.25,.25,.25]
        #             ,[.5,.5,.5,.5,.25,.5,.75,1,1,1,1,.75,.5,.25,.5,.5,.5,.5]
        #             ,[.75,.75,.75,.75,.25,.5,.75,1,1,1,1,.75,.5,.25,.75,.75,.75,.75]
        #             ,[1,1,1,1,.25,.5,.75,1,1,1,1,.75,.5,.25,1,1,1,1]
        #             ,[1,1,1,1,.25,.5,.75,1,1,1,1,.75,.5,.25,1,1,1,1]
        #             ,[1,1,1,1,.25,.5,.75,1,1,1,1,.75,.5,.25,1,1,1,1]
        #             ,[1,1,1,1,.25,.5,.75,1,1,1,1,.75,.5,.25,1,1,1,1]
        #             ,[.5,.5,.5,.5,.25,.5,.75,1,1,1,1,.75,.5,.25,.5,.5,.5,.5]
        #             ,[.75,.75,.75,.75,.25,.5,.75,1,1,1,1,.75,.5,.25,.75,.75,.75,.75]
        #             ,[.25,.25,.25,.25,.25,.5,.75,1,1,1,1,.75,.5,.25,.25,.25,.25,.25]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ,[0,0,0,0,.25,.5,.75,1,1,1,1,.75,.5,.25,0,0,0,0]
        #             ])
        # 
        # cross_edges = cv2.filter2D(edges,-1,kernel)
        # cv2.normalize(cross_edges,cross_edges, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
        col_size = edges.shape[0]
        row_size = edges.shape[1]
        col_data = np.zeros(col_size)
        row_data = np.zeros(row_size)
        for col in range(col_size):
            for row in range(row_size):
                row_data[row] += edges[col][row]
                col_data[col] += edges[col][row]
        
        cross_edges = np.zeros((col_size,row_size),dtype=np.float32)
        for col in range(col_size):
            for row in range(row_size):
                cross_edges[col][row] += col_data[col] + row_data[row]
                
        # cross_edges = non_max_suppression(cross_edges)
        kernel = np.ones((5,5),np.uint8)
        cross_edges = cv2.dilate(cross_edges,kernel,iterations = 1)
        # plt.imshow(cross_edges,cmap = 'gray')
        # plt.show()
        cross_edges = cross_edges*(cross_edges == maximum_filter(cross_edges,footprint=np.ones((150,150))))
        # plt.imshow(cross_edges,cmap = 'gray')
        # plt.show()
        
        return cross_edges
        
    def getEdges(self):
        """
            Return
            ------
                edges : like array
                    this edges
        """
        return self.edges
        
    def getWindows(self):
        """
            Get windows from table image
            using canny edges detection
            
            Return
            ------
                windows : list of image
                    list of window image
        """
        return
        
if __name__ == '__main__':
    
    img = cv2.imread('dataset/full_1.jpg',0)
    table = TableImage(img)
    
    edges = table.getEdges()
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    # windows = table.getWindows()
    
