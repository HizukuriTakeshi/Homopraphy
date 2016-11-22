import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

class FindObject {
	public void run(String pathObject, String pathScene, String pathResult) {

		System.out.println("\nRunning FindObject");

		//-- Step 0: Read the Images
		Mat img_object = Highgui.imread(pathObject, 0); //0 = CV_LOAD_IMAGE_GRAYSCALE
		Mat img_scene = Highgui.imread(pathScene, 0);

		//-- Step 1: Detect the keypoints using SURF Detector
		FeatureDetector detector = FeatureDetector.create(4); //4 = SURF 
		MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
		MatOfKeyPoint keypoints_scene  = new MatOfKeyPoint();
		detector.detect(img_object, keypoints_object);
		detector.detect(img_scene, keypoints_scene);

		//-- Step 2: Calculate descriptors (feature vectors)
		DescriptorExtractor extractor = DescriptorExtractor.create(2); //2 = SURF;
		Mat descriptor_object = new Mat();
		Mat descriptor_scene = new Mat();
		extractor.compute(img_object, keypoints_object, descriptor_object);
		extractor.compute(img_scene, keypoints_scene, descriptor_scene);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		DescriptorMatcher matcher = DescriptorMatcher.create(1); // 1 = FLANNBASED
		MatOfDMatch matches = new MatOfDMatch();
		matcher.match(descriptor_object, descriptor_scene, matches);
		List<DMatch> matchesList = matches.toList();

		Double max_dist = 0.0;
		Double min_dist = 100.0;

		//最小距離
		//-- Quick calculation of max and min distances between keypoints
		for(int i = 0; i < descriptor_object.rows(); i++){
			Double dist = (double) matchesList.get(i).distance;
			if(dist < min_dist) min_dist = dist;
			if(dist > max_dist) max_dist = dist;
		}

		System.out.println("-- Max dist : " + max_dist);
		System.out.println("-- Min dist : " + min_dist);    

		LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
		MatOfDMatch gm = new MatOfDMatch();

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		for(int i = 0; i < descriptor_object.rows(); i++){
			if(matchesList.get(i).distance < 2.5*min_dist){
				good_matches.addLast(matchesList.get(i));
			}
		}

		gm.fromList(good_matches);

		Mat img_matches = new Mat();
		Features2d.drawMatches(
				img_object,
				keypoints_object, 
				img_scene,
				keypoints_scene, 
				gm, 
				img_matches, 
				new Scalar(255,0,0), 
				new Scalar(0,0,255), 
				new MatOfByte(), 
				2);

		//-- Localize the object
		LinkedList<Point> objList = new LinkedList<Point>();
		LinkedList<Point> sceneList = new LinkedList<Point>();

		List<KeyPoint> keypoints_objectList = keypoints_object.toList();
		List<KeyPoint> keypoints_sceneList = keypoints_scene.toList();


		for(int i = 0; i<15/*good_matches.size()*/; i++){
			//-- Get the keypoints from the good matches
			objList.addLast(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
			sceneList.addLast(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
		}

		MatOfPoint2f obj = new MatOfPoint2f();
		obj.fromList(objList);

		MatOfPoint2f scene = new MatOfPoint2f();
		scene.fromList(sceneList);

		Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC, min_dist);
		Mat HP = H.clone();

		if(!H.empty()){
			System.out.println(H.dump());
		}


		//ホモグラフィ変換後の画像の4隅を計算
		Mat P = new Mat(new Size(4,3), CvType.CV_64F);
		P.put(0, 0, new double[] {0,img_object.width(), img_object.width(), 0 ,0, 0, img_object.height(), img_object.height(), 1, 1, 1, 1});
		Mat PP = new Mat(new Size(4,3), CvType.CV_64F);
		//画像の4隅をホモグラフィ変換する
		Core.gemm(H, P, 1, new Mat(), 0, PP);
		
		double x_min,y_min, x_max,y_max;
		Mat dst = new Mat();
		//変換後のx,yの最小値計算
		Core.reduce(PP, dst, 1, Core.REDUCE_MIN);
		x_min = dst.get(0, 0)[0];
		y_min = dst.get(1, 0)[0];
		//変換後のx,yの最大値計算
		Core.reduce(PP, dst, 1, Core.REDUCE_MAX);
		x_max = dst.get(0, 0)[0];
		y_max = dst.get(1, 0)[0];

		double affinex=0, affiney=0;
		//ホモグラフィ変換後の座標が負の領域にある場合、平行移動する
		if(x_min < 0 ){
			HP.put(0, 2, new double[] {HP.get(0, 2)[0]-x_min});
			affinex = -x_min;
		}

		if(y_min < 0 ){
			HP.put(1, 2, new double[] {HP.get(1, 2)[0]-y_min});
			affiney = -y_min;
		}

		//平行移動前の座標
		List<Point> src_pt = new ArrayList<Point>();
		src_pt.add(new Point(0,0));
		src_pt.add(new Point(10,0));
		src_pt.add(new Point(0,10));
		MatOfPoint2f srcpt = new MatOfPoint2f();
		srcpt.fromList(src_pt);
		//平行移動後の座標
		List<Point> dst_pt = new ArrayList<Point>();
		dst_pt.add(new Point(affinex, affiney));
		dst_pt.add(new Point(10+affinex,affiney));
		dst_pt.add(new Point(affinex,10+affiney));
		MatOfPoint2f dstpt = new MatOfPoint2f();
		dstpt.fromList(dst_pt);

		//アフィン行列生成
		Mat A = Imgproc.getAffineTransform(srcpt, dstpt);

		//		Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
		//		Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

		//		obj_corners.put(0, 0, new double[] {0,0});
		//		obj_corners.put(1, 0, new double[] {img_object.cols(),0});
		//		obj_corners.put(2, 0, new double[] {img_scene.cols(),img_scene.rows()});
		//		obj_corners.put(3, 0, new double[] {0,img_scene.rows()});
		//
		//		Core.perspectiveTransform(obj_corners,scene_corners, H);

		//		Core.line(img_object, new Point(scene_corners.get(0,0)), new Point(scene_corners.get(1,0)), new Scalar(0, 255, 0),4);
		//		Core.line(img_object, new Point(scene_corners.get(1,0)), new Point(scene_corners.get(2,0)), new Scalar(0, 255, 0),4);
		//		Core.line(img_object, new Point(scene_corners.get(2,0)), new Point(scene_corners.get(3,0)), new Scalar(0, 255, 0),4);
		//		Core.line(img_object, new Point(scene_corners.get(3,0)), new Point(scene_corners.get(0,0)), new Scalar(0, 255, 0),4);
		//Sauvegarde du résultat

		System.out.println(String.format("Writing %s", pathResult));
		Highgui.imwrite(pathResult, img_matches);


		//カラー画像として読み込む
		Mat img1 = Highgui.imread(pathObject, 0);
		Mat img2 = Highgui.imread(pathScene, 0);

		Mat result = new Mat();
		Mat img_mat = new Mat();

		//入力画像、出力画像、変換行列、サイズ
		Imgproc.warpPerspective(img1, result, HP, new Size(x_max*2,y_max*2));
		Imgproc.warpAffine(img2, img_mat, A, new Size(x_max*2,y_max*2));

		Mat diff = new Mat();
		Core.absdiff(result, img_mat, diff);
		Imgproc.threshold(diff, diff, 100.0, 255.0,Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
		Imgproc.erode(diff, diff, new Mat(), new Point(-1,-1), 1);
		Imgproc.dilate(diff, diff, new Mat());


		//Features2d.drawMatches(img1, keypoints_object, img2, keypoints_scene, gm, img_mat, new Scalar(254,0,0),new Scalar(254,0,0) , new MatOfByte(), 2);
		Highgui.imwrite("resources/imageHomograph.jpg",result);
		Highgui.imwrite("resources/imageAffin.jpg",img_mat);
		Highgui.imwrite("resources/imageDifference.jpg",diff);

	}
}