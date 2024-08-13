%cd ..\; path=cd
%path=uigetdir;
%load(path+"\matlabcode\netlayers_zonepelu_concat.mat",'netlayers');
%----
newIm3ch1=imread(path+"\images\sample1_frame_84.png");
newIm3ch2=imread(path+"\images\sample1_frame_1.png");
%--test1
newIm3ch1=imread('C:\Users\Strouthopoulos Ch\Desktop\myNewFolder\ZONE PELUCIDA SEMANTIC SEGMENTATION\images1\test5.jpg');
newIm3ch1=imresize(newIm3ch1,[256,256]);
newIm3ch2=imresize(newIm3ch2,[256,256]);

newImage1=rgb2gray(newIm3ch1);
newImage2=rgb2gray(newIm3ch2);

%newIm3ch =[newIm3ch1;newIm3ch2];%
newIm3ch =newIm3ch1;

newImage=rgb2gray(newIm3ch);
im = newIm3ch;
%----- test1-----
[CL,scores] = semanticseg(newImage,netlayers);
imzp=(CL=="zonepelu"); %figure; imshow(imzp);

%======= Results Presentation ============================
zpedg = edge(imzp,'sobel');
%figure; imshow(zpedg);

[R, C, ~]=size(im);
for r=1:R
    for c=1:C
        if zpedg(r,c)==true
            im(r,c,1)=255;
            im(r,c,2)=0;
            im(r,c,3)=0;
        end
    end
end

figure;imshow(newIm3ch);
figure;imshow(im)

B = labeloverlay(newImage, CL);
figure;imshow(B)

