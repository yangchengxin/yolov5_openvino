#include "yolov5_infer.h"
#include "Logger.h"
#include "tinyxml2.h"
#include <crtdbg.h>

std::mutex mtx;


InferRequest xml_infer_request;
InferRequest onnx_infer_request;

vector<string>class_names;

string device_name = "CPU";
string model_file = "yolov5s_int8.xml";
string model_file_onnx = "yolo_detect.onnx";
bool load_xml_model = true;
bool xml_model_loading = false;
bool read_onnx = true;
bool read_xml = false;
int detect_num_classes = 0;
//const int overlap = 200;
const int detect_batch = 6;
const float threshold_nms = 0.45;
const float confidence_score = 0.6;
const float threshold_score = 0.25;
int max_detection = 100;

struct Slice_Set
{
	cv::Mat original_frame;
	int col_numbers;
	int row_numbers;
	int w_overlap;
	int h_overlap;
};

 //读取xml文件
std::vector<std::string> fetch_xml_names()
{
	const char* model_xml_path = "yolov5s_int8.xml";
	vector<string> f_coconames;
	int ii = 0;
	//vector<string> f_coconames;
	//mtx.lock();
	tinyxml2::XMLDocument doc;
	// 加载XML文件
	if (doc.LoadFile_(model_xml_path) != tinyxml2::XML_SUCCESS) {
		std::cout << "Failed to load XML file." << std::endl;
		//return 1;
	}

	// 查找names节点
	tinyxml2::XMLElement* names = doc.FirstChildElement("net")->FirstChildElement("rt_info")
		->FirstChildElement("framework")
		->FirstChildElement("names");

	// 获取names节点的value属性值
	const char* namesValue = names->Attribute("value");

	// 输出names值
	//std::cout << "Names: " << namesValue << std::endl;

	std::stringstream ss(namesValue);
	std::string item;
	while (std::getline(ss, item, ','))
	{
		//int key;
		std::string value;
		std::stringstream pairStream(item);
		std::string pairItem;

		if (std::getline(pairStream, pairItem, '['))
		{
			value = pairItem;
		}
		if (std::getline(pairStream, pairItem, '['))
		{
			pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), '\''),
				pairItem.end());
			pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), ' '),
				pairItem.end());
			pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), ']'),
				pairItem.end());
			value = pairItem;
		}
		f_coconames.push_back(value);
	}
	for (int ii = 0; ii < f_coconames.size(); ii++)
	{
		if (ii == 0)
		{
			class_names.push_back(f_coconames[ii]);
		}
		else if (ii > 0 && ii < f_coconames.size() - 1)
		{
			int ss = f_coconames[ii].size();
			string coconame_ = f_coconames[ii].substr(2, ss - 3);
			class_names.push_back(coconame_);
		}
		else if (ii == f_coconames.size() - 1)
		{
			int ss = f_coconames[ii].size();
			string coconame_ = f_coconames[ii].substr(2, ss - 4);
			class_names.push_back(coconame_);
		}
	}
	return class_names;
}

// 读取onnx文件
std::vector<std::string> fetch_onnx_names()
{
	const wchar_t* model_path = L"yolo_detect.onnx";

	vector<string> f_coconames;
	vector<string> f_class_names;
	//读取onnx中的类别信息
	//0、实例化一个ort env
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	//1、实例化一个session options
	Ort::SessionOptions session_options;

	//set op threads
	session_options.SetIntraOpNumThreads(1);

	//set optimization options
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	//2、实例化一个session对象
	Ort::Session session(env, model_path, session_options);

	//memory allocation and options
	Ort::AllocatorWithDefaultOptions allocator;
	auto model_metadata = session.GetModelMetadata();
	auto custom_metadata_map_keys =
		model_metadata.GetCustomMetadataMapKeysAllocated(allocator);
	//std::cout << "Model Metadata: " << std::endl;
	for (auto& key : custom_metadata_map_keys)
	{
		std::string key_str = key.get();
		std::string value_str =
			model_metadata
			.LookupCustomMetadataMapAllocated(key_str.c_str(), allocator)
			.get();
		//std::cout << "key: " << key_str << " value: " << value_str << std::endl;
		if (key_str == "names")
		{
			std::stringstream ss(value_str);
			std::string item;
			while (std::getline(ss, item, ','))
			{
				//int key;
				std::string value;
				std::stringstream pairStream(item);
				std::string pairItem;

				if (std::getline(pairStream, pairItem, '['))
				{
					value = pairItem;
				}
				if (std::getline(pairStream, pairItem, '['))
				{
					pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), '\''),
						pairItem.end());
					pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), ' '),
						pairItem.end());
					pairItem.erase(std::remove(pairItem.begin(), pairItem.end(), ']'),
						pairItem.end());
					value = pairItem;
				}
				f_coconames.push_back(value);
			}
		}
	}
	for (int ii = 0; ii < f_coconames.size(); ii++)
	{
		if (ii == 0)
		{
			f_class_names.push_back(f_coconames[ii]);
		}
		else if (ii > 0 && ii < f_coconames.size() - 1)
		{
			int ss = f_coconames[ii].size();
			string coconame_ = f_coconames[ii].substr(2, ss - 3);
			f_class_names.push_back(coconame_);
		}
		else if (ii == f_coconames.size() - 1)
		{
			int ss = f_coconames[ii].size();
			string coconame_ = f_coconames[ii].substr(2, ss - 4);
			f_class_names.push_back(coconame_);
		}
	}
	return f_class_names;
}

void Initial_xml_model()
{
	if (load_xml_model && xml_model_loading == false)
	{
		Core core;
		CompiledModel compiled_model = core.compile_model(model_file, device_name);
		if (compiled_model)
		{
			xml_infer_request = compiled_model.create_infer_request();
		}
		if (xml_infer_request)
		{
			xml_model_loading = true;
		}
	}
}

void Init_xml()
{
	Initial_xml_model();
}

void Init_onnx()
{
	LOG_PRINT_INFO("start initial");
	Core core;
	CompiledModel compiled_onnx_model = core.compile_model(model_file_onnx, device_name);
	onnx_infer_request = compiled_onnx_model.create_infer_request();
	LOG_PRINT_INFO("finish initial");
}

CPPDLL_API void Initial_model()
{
	Init_onnx();
	if (read_xml)
	{
		class_names = fetch_xml_names();
		read_onnx = true;
	}
	else if (read_onnx)
	{
		class_names = fetch_onnx_names();
	}
	detect_num_classes = class_names.size();
	string detect_num_classes_string = to_string(detect_num_classes);
	LOG_PRINT_INFO("detect class size: %s", detect_num_classes_string);

	thread xml_thread(Init_xml);
	xml_thread.detach();
}

void detect(cv::Mat frame, string& return_result, const int delta_x, const int delta_y, cv::Mat original_frame)
{
	Shape tensor_shape;
	Tensor input_node;
	if (xml_model_loading)
	{
		if (onnx_infer_request)
		{
			onnx_infer_request.~InferRequest();
		}
		input_node = xml_infer_request.get_input_tensor();
		tensor_shape = input_node.get_shape();

		int w = frame.cols;
		int h = frame.rows;
		int _max = max(h, w);
		cv::Mat image(Size(_max, _max), CV_8UC3, cv::Scalar(255, 255, 255));
		cv::Rect roi(0, 0, w, h);
		frame.copyTo(image(roi));
		//交换RB通道
		cvtColor(image, image, COLOR_BGR2RGB);
		//计算缩放因子
		size_t num_channels = tensor_shape[1];
		size_t height = tensor_shape[2];
		size_t width = tensor_shape[3];
		float x_factor = image.cols / float(width);
		float y_factor = image.rows / float(height);

		//缩放图片并归一化
		cv::Mat blob_image;
		cv::resize(image, blob_image, cv::Size(width, height));
		blob_image.convertTo(blob_image, CV_32F);
		blob_image = blob_image / 255.0;

		//4.将图像数据填入input_tensor
		Tensor input_tensor = xml_infer_request.get_input_tensor();
		//创建指向模型输入节点的指针
		float* input_tensor_data = input_node.data<float>();
		// 将图片数据填充到模型输入节点中
		// 原有图片数据为 HWC格式，模型输入节点要求的为 CHW 格式
		/*std::vector<cv::Mat> planes(3);
		for (size_t pId = 0; pId < planes.size(); pId++)
		{
			planes[pId] = cv::Mat(cv::Size(width, height), CV_32FC1, input_tensor_data + pId * cv::Size(width, height).area());
		}
		cv::split(blob_image, planes);*/

		for (size_t c = 0; c < num_channels; c++) {
			for (size_t h = 0; h < height; h++) {
				for (size_t w = 0; w < width; w++) {
					input_tensor_data[c * width * height + h * width + w] = blob_image.at<Vec<float, 3>>(h, w)[c];
				}
			}
		}

		// 5.执行推理计算
		auto start1 = std::chrono::system_clock::now();
		xml_infer_request.infer();
		auto end1 = std::chrono::system_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
		//cout << "left_up推理耗时:" << duration << "ms" << endl;

		// 6.处理推理计算结果
		// 6.1 获得推理结果
		const ov::Tensor& output = xml_infer_request.get_tensor("output");
		const float* output_buffer = output.data<const float>();

		// 6.2 解析推理结果，YOLOv5 output format: cx,cy,w,h,score
		int out_rows = output.get_shape()[1]; //获得"output"节点的rows
		int out_cols = output.get_shape()[2]; //获得"output"节点的cols
		Mat det_output(out_rows, out_cols, CV_32F, (float*)output_buffer);

		vector<cv::Rect> boxes;
		vector<int> classIds;
		vector<float> confidences;

		for (int i = 0; i < det_output.rows; i++) {
			float confidence = det_output.at<float>(i, 4);
			if (confidence < confidence_score) {
				continue;
			}
			int endindex = detect_num_classes + 5;
			Mat classes_scores = det_output.row(i).colRange(5, endindex);
			Point classIdPoint;
			double score;
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

			// 置信度 0～1之间
			if (score > confidence_score)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
				int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
				int width = static_cast<int>(ow * x_factor);
				int height = static_cast<int>(oh * y_factor);
				Rect box;
				box.x = x;
				box.y = y;
				box.width = width;
				box.height = height;

				boxes.push_back(box);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}

		// NMS
		vector<int> indexes;
		dnn::NMSBoxes(boxes, confidences, threshold_score, threshold_nms, indexes);
		int count = 0;
		if (indexes.size() > max_detection) {
			count = max_detection;
		}
		else {
			count = indexes.size();
		}
		for (size_t i = 0; i < count; i++) {
			int index = indexes[i];
			int idx = classIds[index];


			int x0 = int(boxes[index].x);
			int y0 = int(boxes[index].y);
			int x1 = int(boxes[index].x) + int(boxes[index].width);
			int y1 = int(boxes[index].y) + int(boxes[index].height);
			int area = boxes[index].width * boxes[index].height;


			x0 += delta_x;
			x1 += delta_x;
			y0 += delta_y;
			y1 += delta_y;
			return_result += class_names[idx] + "," + std::to_string(int(x0 * 4)) + "," + std::to_string(int(y0 * 4)) + "," + std::to_string(int(x1 * 4)) + "," +
				std::to_string(int(y1 * 4)) + "," + std::to_string(int(area)) + "," + std::to_string(confidences[index]) + ";" + "\n";

			rectangle(original_frame, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 0), 2);

			//cout << " " << endl;

			putText(original_frame, class_names[idx], Point(x0, y0 - 10), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 0, 0));
		}
		/*cv::namedWindow("int8", WINDOW_NORMAL);
		cv::imshow("int8", original_frame);
		cv::waitKey(0);*/
		//cv::imwrite("fast_track_1\\test.bmp", original_frame);
	}
	else
	{
		input_node = onnx_infer_request.get_input_tensor();
		tensor_shape = input_node.get_shape();

		//Mat frame = imread(".\\test_data0\\2.png", IMREAD_COLOR);
	//Lettterbox resize is the default resize method in YOLOv5.
		int w = frame.cols;
		int h = frame.rows;
		int _max = max(h, w);
		Mat image(Size(_max, _max), CV_8UC3, cv::Scalar(255, 255, 255));
		Rect roi(0, 0, w, h);
		frame.copyTo(image(roi));
		//交换RB通道
		cvtColor(image, image, COLOR_BGR2RGB);
		//计算缩放因子
		size_t num_channels = tensor_shape[1];
		size_t height = tensor_shape[2];
		size_t width = tensor_shape[3];
		float x_factor = image.cols / float(width);
		float y_factor = image.rows / float(height);

		//int64 start = cv::getTickCount();
		//缩放图片并归一化
		Mat blob_image;
		resize(image, blob_image, cv::Size(width, height));
		blob_image.convertTo(blob_image, CV_32F);
		blob_image = blob_image / 255.0;

		// 4.3 将图像数据填入input tensor
		Tensor input_tensor = onnx_infer_request.get_input_tensor();
		// 获取指向模型输入节点数据块的指针
		float* input_tensor_data = input_node.data<float>();
		// 将图片数据填充到模型输入节点中
		// 原有图片数据为 HWC格式，模型输入节点要求的为 CHW 格式
		for (size_t c = 0; c < num_channels; c++) {
			for (size_t h = 0; h < height; h++) {
				for (size_t w = 0; w < width; w++) {
					input_tensor_data[c * width * height + h * width + w] = blob_image.at<Vec<float, 3>>(h, w)[c];
				}
			}
		}

		// 5.执行推理计算
		auto start1 = std::chrono::system_clock::now();
		onnx_infer_request.infer();
		auto end1 = std::chrono::system_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
		//cout << "left_up推理耗时:" << duration << "ms" << endl;


		// 6.处理推理计算结果
		// 6.1 获得推理结果
		const ov::Tensor& output = onnx_infer_request.get_tensor("output");
		const float* output_buffer = output.data<const float>();

		// 6.2 解析推理结果，YOLOv5 output format: cx,cy,w,h,score
		int out_rows = output.get_shape()[1]; //获得"output"节点的rows
		int out_cols = output.get_shape()[2]; //获得"output"节点的cols
		Mat det_output(out_rows, out_cols, CV_32F, (float*)output_buffer);

		vector<cv::Rect> boxes;
		vector<int> classIds;
		vector<float> confidences;

		for (int i = 0; i < det_output.rows; i++) {
			float confidence = det_output.at<float>(i, 4);
			if (confidence < confidence_score) {
				continue;
			}
			int endindex = detect_num_classes + 5;
			Mat classes_scores = det_output.row(i).colRange(5, endindex);
			Point classIdPoint;
			double score;
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

			// 置信度 0～1之间
			if (score > confidence_score)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
				int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
				int width = static_cast<int>(ow * x_factor);
				int height = static_cast<int>(oh * y_factor);
				Rect box;
				box.x = x;
				box.y = y;
				box.width = width;
				box.height = height;

				boxes.push_back(box);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}


		// NMS
		vector<int> indexes;
		dnn::NMSBoxes(boxes, confidences, threshold_score, threshold_nms, indexes);
		int count = 0;
		if (indexes.size() > max_detection) {
			count = max_detection;
		}
		else {
			count = indexes.size();
		}
		for (size_t i = 0; i < count; i++) {
			int index = indexes[i];
			int idx = classIds[index];


			int x0 = int(boxes[index].x);
			int y0 = int(boxes[index].y);
			int x1 = int(boxes[index].x) + int(boxes[index].width);
			int y1 = int(boxes[index].y) + int(boxes[index].height);
			int area = boxes[index].width * boxes[index].height;
			//cout << "x0=" << x0 << ",x1=" << x1 << ",y0=" << y0 << ",y1=" << y1 << endl;


			x0 += delta_x;
			x1 += delta_x;
			y0 += delta_y;
			y1 += delta_y;
		
		/*	cout << "转换完之后：" << endl;
			cout << "x0=" << x0 << ",x1=" << x1 << ",y0=" << y0 << ",y1=" << y1 << endl << endl;*/
			return_result += class_names[idx] + "," + std::to_string(int(x0 * 4)) + "," + std::to_string(int(y0 * 4)) + "," + std::to_string(int(x1 * 4)) + "," +
				std::to_string(int(y1 * 4)) + "," + std::to_string(int(area)) + "," + std::to_string(confidences[index]) + ";" + "\n";

			rectangle(original_frame, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 255), 4);
			//onnx_detect_count++;

			//cout << "onnx_detect_count = " << onnx_detect_count << endl;
			//cout << " " << endl;

			putText(original_frame, class_names[idx], Point(x0, y0 - 10), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 0, 0));
		}
		/*cv::namedWindow("onnx", WINDOW_NORMAL);
		cv::imshow("onnx", original_frame);
		cv::waitKey(0);*/
	}
}

SliceImageData Slice_Image(cv::Mat frame, const int col_numbers, const int row_numbers, const int w_overlap, const int h_overlap)
{
	SliceImageData Slice_data;
	//if (int(col_numbers * row_numbers) % 2 != 0)
	//{
	//	cout << "请输入2的倍数!" << endl;
	//	exit(0);
	//}
	int w = frame.cols;//待切图的宽度
	int h = frame.rows;//待切图的高度
	int sub_width = w / col_numbers;//切图后每一块的宽度
	int sub_height = h / row_numbers;//切图后每一块的高度
	vector<cv::Mat> image_batch;
	if (col_numbers != 1 || row_numbers != 1)
	{
		for (int yy = 0; yy < row_numbers; yy++)
		{
			for (int xx = 0; xx < col_numbers; xx++)
			{
				cv::Mat src;
				if (xx != 0 && yy != 0)
				{
					frame({ sub_width * xx - w_overlap, sub_height * yy - h_overlap, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					image_batch.push_back(src);
				}
				else if (xx == 0 && yy != 0)
				{
					frame({ 0, sub_height * yy - h_overlap, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					image_batch.push_back(src);
				}
				else if (xx != 0 && yy == 0)
				{
					frame({ sub_width * xx - w_overlap, 0, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					image_batch.push_back(src);
				}
				else if (xx == 0 && yy == 0)
				{
					frame({ 0, 0, sub_width + w_overlap, sub_height + h_overlap }).copyTo(src);
					image_batch.push_back(src);
				}
			}
		}
	}
	image_batch.push_back(frame);
	Slice_data.Original_Frame = frame;
	Slice_data.Slice_Batch = image_batch;
	Slice_data.Slice_Cols = col_numbers;
	Slice_data.Slice_Rows = row_numbers;
	Slice_data.Slice_width = sub_width;
	Slice_data.Slice_height = sub_height;
	Slice_data.Slice_Overlap_W = w_overlap;
	Slice_data.Slice_Overlap_H = h_overlap;
	return Slice_data;
}

CPPDLL_API int Detect(uchar* input_ptr, int height, int width, int channels, int col_nums, int row_nums, int w_overlap, int h_overlap, char*& output)
{
	/* 畸变矫正 */
	cv::Mat inputcalib = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	uchar* inputcalib_ptr = inputcalib.data;
	Calibrate(input_ptr, height, width, inputcalib_ptr);

	cv::Mat frame = cv::Mat(height, width, CV_8UC3);
	if (channels == 3)
	{
		if (inputcalib_ptr != nullptr)
		{
			frame = cv::Mat(height, width, CV_8UC3, inputcalib_ptr);
		}
		else
		{
			cout << "ptr is null" << endl;
			return -1;
		}
	}
	else if (channels == 1 || channels == 2)
	{
		return -1;
	}
	//string detect_result;
	string result;
	cv::pyrDown(frame, frame);
	cv::pyrDown(frame, frame);
	/*int width = frame.cols;
	int height = frame.rows;*/
	
	//设置分图参数
	Slice_Set slice_set;
	slice_set.original_frame = frame;
	slice_set.col_numbers = col_nums;
	slice_set.row_numbers = row_nums;
	slice_set.w_overlap = w_overlap;
	slice_set.h_overlap = h_overlap;

	SliceImageData slice_data;
	slice_data = Slice_Image(slice_set.original_frame, slice_set.col_numbers, slice_set.row_numbers, slice_set.w_overlap, slice_set.h_overlap);
	//for (int i_ = 0; i_ < slice_data.Slice_Batch.size(); i_++)
	//{
	//	cv::Mat slice_show = slice_data.Slice_Batch[i_];
	//	string str_i = to_string(i_);
	//	cv::namedWindow("切片展示" + str_i);
	//	cv::imshow("切片展示" + str_i, slice_show);
	//	cv::waitKey(0);
	//}
	auto start_time = std::chrono::high_resolution_clock::now();
	for (int ii = 0; ii < slice_data.Slice_Batch.size(); ii++)
	{
		int Judge_col, Judge_row;
		Judge_row = ii / slice_data.Slice_Cols; //等于0的话说明是第一行，不等于0的话就是这个数是几就是第 judge_row + 1 行
		Judge_col = ii % slice_data.Slice_Cols; //在Judge_col不等于0的情况下，可以通过这个值判断第几列
		//cout << "Judge_row = " << Judge_row << endl;
		//cout << "Judge_col = " << Judge_col << endl;
		string detect_result;
		/*第一行第一个*/
		if (Judge_col == 0 && Judge_row == 0 && ii != slice_data.Slice_Batch.size() - 1)
		{
			detect(slice_data.Slice_Batch[ii], detect_result, 0, 0, slice_data.Original_Frame);
		}
		/*第一列除了第一个*/
		else if (Judge_col == 0 && Judge_row != 0 && ii != slice_data.Slice_Batch.size() - 1)
		{
			detect(slice_data.Slice_Batch[ii], detect_result, 0, slice_data.Slice_height * Judge_row - slice_data.Slice_Overlap_H, slice_data.Original_Frame);
		}
		/*第一行除了第一个*/
		else if (Judge_col != 0 && Judge_row == 0 && ii != slice_data.Slice_Batch.size() - 1)
		{
			detect(slice_data.Slice_Batch[ii], detect_result, slice_data.Slice_width * Judge_col - slice_data.Slice_Overlap_W, 0, slice_data.Original_Frame);
		}
		/*剩下的切图*/
		else if (Judge_col !=0 && Judge_row != 0 && ii != slice_data.Slice_Batch.size() - 1)
		{
			detect(slice_data.Slice_Batch[ii], detect_result, slice_data.Slice_width * Judge_col - slice_data.Slice_Overlap_W, slice_data.Slice_height * Judge_row - slice_data.Slice_Overlap_H, slice_data.Original_Frame);
		}
		/*最后的一张整图*/
		else if (ii == slice_data.Slice_Batch.size() - 1)
		{
			detect(slice_data.Slice_Batch[ii], detect_result, 0, 0, slice_data.Original_Frame);
		}
		result += detect_result;
	}
	char out[10000];
	int jj;
	for (jj = 0; jj < result.length(); jj++)
	{
		int LL = result.length();
		string SSS = to_string(result[jj]);
		// 分号的ascII码是59
		if (to_string(result[jj]) == "59" && jj == result.length() - 2)
		{
			continue;
		}
		out[jj] = result[jj];
	}
	out[jj] = '\0';
	output = out;

	auto end_time = std::chrono::high_resolution_clock::now();
	float infer_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
	std::cout << "infer time=" << infer_time << std::endl;
	return 1;
	//_CrtDumpMemoryLeaks();
}