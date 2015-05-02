function output = findCircles(img, scale)
    smallImg = imresize(img, .3);
    [centers, radii] = imfindcircles(smallImg,[2 50],'ObjectPolarity','bright', 'Sensitivity', .9);
    radii = radii/.3/scale;
    centers = centers/.3/scale;
    output = struct('centers', centers, 'radii', radii);
end

