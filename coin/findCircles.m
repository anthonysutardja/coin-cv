img = imread('asd.png');
[c, r] = imfindcircles(img,[2 200],'ObjectPolarity','bright', 'Sensitivity', .9);
imshow(img);
h = viscircles(c,r);