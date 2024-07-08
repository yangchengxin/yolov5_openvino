#include "calibrator.h"

CPPDLL_API void Calibrate(uchar* input_image, int rows, int cols, uchar* output_image)
{
	/* 读取yaml文件来获取标定的参数（获取cameraMatrix，distCoeffs参数） */
	cv::Mat cameraMatrix = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
	cv::Mat distCoeffs	 = cv::Mat(1, 5, CV_64F, cv::Scalar(0));

	/* 标定的参数文件 yaml */
	std::string yaml_path = camera_config + yaml_name;
	cv::FileStorage ff(yaml_path, cv::FileStorage::READ);
	if (!ff.isOpened())
	{
		std::cout << "标定配置文件读取失败" << std::endl;
	}
	ff["camera_matrix"] >> cameraMatrix;
	ff["camera_distCoeffs"] >> distCoeffs;
	ff.release();

	/* 通过读取的标定的参数来对输入图像经行矫正 */
	cv::Size image_size(cols, rows);

	/* 定义畸变矫正的输入参数，映射矩阵，旋转矩阵 */
	cv::Mat mapx = cv::Mat(image_size, CV_32FC1);
	cv::Mat mapy = cv::Mat(image_size, CV_32FC1);
	cv::Mat R	 = cv::Mat::eye(3, 3, CV_32F);

	/* 计算映射矩阵 */
	cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
	cv::Mat Input_Image  = cv::Mat(image_size.height, image_size.width, CV_8UC3, input_image);
	/* 这里需要动态分配一下内存给矫正后的图像矩阵，因为如果是局部变量的话，在退出函数时会自动销毁变量，返回的指针是野指针 */
	//cv::Mat* Output_Image = new cv::Mat(image_size.height, image_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat Output_Image = cv::Mat(image_size.height, image_size.width, CV_8UC3, cv::Scalar(0,0,0));

	cv::remap(Input_Image, Output_Image, mapx, mapy, cv::INTER_LINEAR);

	/* 动态分配内存法，还需要将output_image改为引用的方式传入，uchar*& output_image */
	//output_image = Output_Image->data;
	std::memcpy(output_image, Output_Image.data, Output_Image.channels() * Output_Image.rows * Output_Image.cols * sizeof(uint8_t));
}

//int main(int argc, char* argv[])
//{
//	cv::String image_path = "1.bmp";
//	cv::Mat image		  = cv::imread(image_path);
//	int rows			  = image.rows;
//	int cols			  = image.cols;
//	uchar* input_ptr	  = image.data;
//
//	/* 创建一个模板mat矩阵 */
//	cv::Mat output_ = cv::Mat(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
//	/* 获取该模板矩阵的指针 */
//	uchar* output_ptr = output_.data;
//	Calibrate(input_ptr, rows, cols, output_ptr);
//	if (output_ptr != nullptr)
//	{
//		cv::Mat output = cv::Mat(rows, cols, CV_8UC3, output_ptr);
//		cv::namedWindow("calib", cv::WINDOW_NORMAL);
//		cv::imshow("calib", output);
//	}
//	cv::namedWindow("no_calib", cv::WINDOW_NORMAL);
//	cv::imshow("no_calib", image);
//	cv::waitKey(0);
//
//	//delete output_ptr;
//	return 0;
//}