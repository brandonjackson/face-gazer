# this code should be used in future versions of this software
# it was not implemented because it did not add to the presentation
# but it allows for dynamic cropping and radii estimations when
# a more sophisticated scheme is put in place.  It can go directly into
# the main while loop and the i



        # keep track of pupil centers for automatic frame cropping later
        # note: the array eyellipse stores all the points the pupil center
        # ever occupies but is never made use of.  Future users of this
        # code should make use of this data if only to crop the region of
        # interest. 
        if eyedata_initialized == False:
                eyellipse = np.ones(frames['eye'].shape)*255;
                eyellipse = eyellipse.astype(np.uint8);
                eyedata_initialized = True

	# use data accumulated during calibration constrain ROI
        # and radius for pupil
        # if i >= 20 and done_accum is False:
        #       if gotedgePoints == True:
        #               # add the center to eyePoints array and display
        #               # accumulated points
        #               if eyePoints_initialized is True:
        #                       eyePoints = np.vstack(\
        #                               [eyePoints, [center[1],center[0]]]);
        #                       eyellipse[center[1],center[0]] = 0;
        #                       cv2.imshow("eyellipse",eyellipse);
        #               else:
        #                       eyePoints = np.array(\
        #                               [[center[1],center[0]]]);
        #                       eyePoints_initialized = True;   

        #       # if there are enough accumulated points to fit an ellipse
        #       # fit an ellipse to the points
        #       if eyePoints_initialized is True and eyePoints.shape[0] > 6:
        #               ellipseBox = cv2.fitEllipse(eyePoints);
        #               eBox = tuple([tuple([ellipseBox[0][1],\
        #               ellipseBox[0][0]]),tuple([ellipseBox[1][1],\
        #               ellipseBox[1][0]]),ellipseBox[2]*-1]);
        #               cv2.ellipse(ellipseFrame,eBox,(0, 0, 255));
        #               cv2.imshow("ellipseFit",ellipseFrame);

        #       # if calibration is done then accumulation is done
        #       if done:
        #               done_accum = True;


        #               # show the elipse in a new image frame
        #               dat = np.ones(frames['eye'].shape);
        #               dat = dat.astype(np.uint8);
        #               cv2.ellipse(dat,eBox,0);
        #               cv2.ellipse(eyellipse,eBox,0);
        #               cv2.imshow("eyellipse",eyellipse);

        #               # find the points on the elipse and construce a
        #               # upright bounding box around them
        #               edgePoints = np.argwhere(dat==0);
        #               maxvals = np.amax(edgePoints, axis=0);
        #               minvals = np.amin(edgePoints, axis=0);
        #               print minvals;
        #               print maxvals;  
        #               height = maxvals[1]-minvals[1];
        #               width = maxvals[0]-minvals[0];
        #               newheight = height*2;
        #               newwidth = width*2;
        #               newminvals = (minvals - minvals/2).astype(int); 
        #               newmaxvals = minvals + [newheight,newwidth];

        #               print newminvals;
        #               print newmaxvals;

        #               scaledxmin = newminvals[0]; 
        #               scaledymin = newminvals[1]; 
        #               scaledxmax = newmaxvals[0]; 
        #               scaledymax = newmaxvals[1]; 
        #               #contours, hierarchy = cv2.findContours(dat,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
        #               #bBox = cv2.boundingRect(contours[0]);  
        #               #cv2.rectangle(eyellipse,(minvals[1],minvals[0]),\
        #               #       (maxvals[1],maxvals[0]),20);
        #               cv2.imshow("eyellipse",eyellipse);




