#include "yolov5_infer.h"
//#include"Logger.h"
//#include<mutex>
std::mutex mtx_1;

const static bool slice_image = true;
const static bool run_detect  = false;
const static int slice_cols = 1;
const static int slice_rows = 1;

int main()
{
	if(run_detect)
	{
		try 
		{
			cout << "1111111111111111111" << endl;
			auto start_time00 = std::chrono::system_clock::now();
			vector<thread> mythreads;

			//模拟软件那边的多线程调用初始化模型
			//for (int i = 0; i < 2; i++)
			//{
			//	mythreads.push_back(thread(Initial_model));
			//}
			//for (auto iter = mythreads.begin(); iter != mythreads.end(); ++iter)
			//{
			//	iter->join();
			//}

			Initial_model();
			auto start_time11 = std::chrono::system_clock::now();
			std::chrono::duration<double> diff = start_time11 - start_time00;;
			std::cout << "模型加载时间：" << diff.count() << "s" << std::endl;

			std::vector<cv::String> filenames;
			cv::String folder = "ImagesAndVideos\\*.bmp";
			cv::glob(folder, filenames);
			while (true)
			{
				for (size_t i = 0; i < filenames.size(); i++)
				{
					cv::Mat image = cv::imread(filenames[i]);
					int height_ = image.rows;
					int width_ = image.cols;
					int channels = image.channels();
					int w_overlap = 0;
					int h_overlap = 0;
					cout << "当前打开的图像路径： " << filenames[i] << endl;
					uchar* input_ptr = image.data;
					//cv::resize(image, image, cv::Size(2048, 3000));
					char* output = NULL;
					auto infer_time00 = std::chrono::system_clock::now();
					int a = Detect(input_ptr, height_, width_, channels, slice_cols, slice_rows, w_overlap, h_overlap, output);
					auto infer_time11 = std::chrono::system_clock::now();
					std::chrono::duration<double> diff = infer_time11 - infer_time00;
					std::cout << "解析检测结果： " << endl << output;
					std::cout << "检测耗时： " << diff.count() << "s" << std::endl << std::endl;
					output = NULL;
				}
			}
		}
		catch (const std::exception& ex) {
			std::cerr << ex.what() << std::endl;
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}
	if (slice_image)
	{
		int slice_cols = 0, slice_rows = 0;
		cv::String folder_input;
		cv::String folder_output;

		std::cout << "请输入被切图的路径： ";
		std::cin >> folder_input;
		std::cout << std::endl;

		std::cout << "请输入需要切图的行数： ";
		std::cin >> slice_rows;
		std::cout << std::endl;

		std::cout << "请输入需要切图的列数： ";
		std::cin >> slice_cols;
		std::cout << std::endl;

		std::cout << "请输入切图后保存的路径： ";
		std::cin >> folder_output;
		std::cout << std::endl;

		const int overlap = 100;
		std::vector<cv::String> filenames;
		//cv::String folder = "427\\*.bmp";
		cv::glob(folder_input, filenames);
		for (int i = 0; i < filenames.size();i++)
		{
			cv::Mat image = cv::imread(filenames[i]);
			int w_overlap = 100;
			int h_overlap = 100;
			cout << "当前打开的图像路径： " << filenames[i] << endl;
			SliceImageData sliceimagedata = Slice_Image(image, slice_cols, slice_rows, overlap, overlap);
			for (int j = 0; j < sliceimagedata.Slice_Batch.size(); j++)
			{
				string ii = to_string(i);
				string jj = to_string(j);
				//cv::String folder = "427_slice";
				cv::String image_path = folder_output + '\\' + ii + '_' + jj + ".bmp";
				cv::imwrite(image_path, sliceimagedata.Slice_Batch[j]);
			}
		}
	}
	system("pause");
	return 0;
}