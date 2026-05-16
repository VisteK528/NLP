
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_community.chat_models import ChatOllama
from datasets import Dataset
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from google.colab import userdata
from ragas.embeddings import embedding_factory


data = {
    "question": [
    "What does cv::ximgproc::anisotropicDiffusion do, and what is the alpha parameter?",
    "What does cv::ximgproc::edgePreservingFilter do and what is the d parameter?",
    "What thresholding technique does cv::ximgproc::niBlackThreshold use?",
    "How does cv::cuda::LookUpTable::transform work?",
    "What does cv::FaceDetectorYN::setNMSThreshold do?",
    "What does cv::FaceDetectorYN::setScoreThreshold do?",
    "What does cv::MinProblemSolver::minimize do and what is the x parameter?",
    "What is cv::setOpenGlDrawCallback used for?",
    "What does cv::adaptiveThreshold do?",
    "What does cv::HoughLines do and what does the lines output contain?",
    "What does open3d.geometry.PointCloud.voxel_down_sample do?",
    "What algorithm does open3d.geometry.PointCloud.cluster_dbscan use, and what does a label of -1 indicate?",
    "What algorithm does open3d.t.geometry.PointCloud.segment_plane use?",
    "What does open3d.geometry.PointCloud.estimate_normals do?",
    "What does open3d.visualization.draw_geometries do?",
    "What does open3d.geometry.PointCloud.compute_point_cloud_distance compute?",
    "What does open3d.camera.PinholeCameraIntrinsic store?",
    "What is open3d.geometry.AxisAlignedBoundingBox and how is it generated?",
    "What does open3d.t.geometry.PointCloud.remove_statistical_outliers do?",
    "What does open3d.core.addmm compute?",
    "What does pcl::ApproximateVoxelGrid::setLeafSize do?",
    "What does pcl::StatisticalOutlierRemoval::setMeanK do?",
    "What does pcl::SACSegmentation::setMethodType do?",
    "What does pcl::PassThrough::setFilterFieldName do?",
    "What does pcl::gpu::EuclideanClusterExtraction::setClusterTolerance do?",
    "What does pcl::FPFHEstimation::computeFeature estimate?",
    "What does pcl::registration::TransformationEstimation estimate?",
    "What does pcl::RadiusOutlierRemoval::setMinNeighborsInRadius do?",
    "What does pcl::filters::Convolution3D::setRadiusSearch do?",
    "What does pcl::gpu::EuclideanClusterExtraction::extract do?"
  ],
  "ground_truth": [
    "cv::ximgproc::anisotropicDiffusion performs anisotropic diffusion on an image using the Perona-Malik method, which is the solution to a partial differential equation. It requires a source image with 3 channels and writes the result to a destination image of the same size. The alpha parameter controls the amount of time step per iteration.",
    "cv::ximgproc::edgePreservingFilter smoothes an image using the Edge-Preserving filter. It reduces both Gaussian noise and salt & pepper noise. The source must be an 8-bit 3-channel image. The d parameter is the diameter of each pixel neighborhood that is used during filtering.",
    "cv::ximgproc::niBlackThreshold performs thresholding using Niblack's technique or popular variations inspired by it. The function transforms a grayscale image to a binary image. It accepts an 8-bit single-channel source image and produces a destination image of the same size and type.",
    "cv::cuda::LookUpTable::transform transforms the source matrix into the destination matrix using the given look-up table, applying the formula dst(I) = lut(src(I)). Currently, CV_8UC1 and CV_8UC3 source matrices are supported. It also accepts an optional stream for asynchronous operation.",
    "cv::FaceDetectorYN::setNMSThreshold sets the Non-maximum-suppression threshold to suppress bounding boxes that have an IoU greater than the given value. The parameter nms_threshold specifies the threshold for the NMS operation.",
    "cv::FaceDetectorYN::setScoreThreshold sets the score threshold to filter out bounding boxes whose score is less than the given value. The parameter score_threshold specifies the threshold for filtering out bounding boxes.",
    "cv::MinProblemSolver::minimize actually runs the minimization algorithm. The x parameter is the initial point that becomes the centroid of the initial simplex. After the algorithm terminates, x will be set to the point where the minimum was found. The function returns the minimum value of the objective function.",
    "cv::setOpenGlDrawCallback sets a callback function to be called to draw on top of the displayed image. It can be used to draw 3D data on the window. The winname parameter specifies the target window, and onOpenGlDraw is a pointer to the function to be called every frame.",
    "cv::adaptiveThreshold applies an adaptive threshold to an array, transforming a grayscale image to a binary image. The function can process the image in-place. It requires an 8-bit single-channel source image and produces a destination image of the same size and type.",
    "cv::HoughLines finds lines in a binary image using the standard Hough transform algorithm. It requires an 8-bit single-channel binary source image. The lines parameter is an output vector where each line is represented by its detected parameters. The image may be modified by the function during processing.",
    "open3d.geometry.PointCloud.voxel_down_sample downsamples the input point cloud into an output point cloud using a voxel grid. If normals and colors exist in the input, they are averaged within each voxel.",
    "open3d.geometry.PointCloud.cluster_dbscan clusters a PointCloud using the DBSCAN algorithm by Ester et al. (1996), 'A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise'. It returns a list of point labels, where a label of -1 indicates noise according to the algorithm.",
    "open3d.t.geometry.PointCloud.segment_plane segments a plane in the point cloud using the RANSAC algorithm. It is a wrapper for a CPU implementation, and a copy of the point cloud data as well as the resulting plane model and inlier indices will be made.",
    "open3d.geometry.PointCloud.estimate_normals computes the normals of a point cloud. Normals are oriented with respect to the input point cloud if normals already exist in it.",
    "open3d.visualization.draw_geometries draws a list of geometry objects.",
    "open3d.geometry.PointCloud.compute_point_cloud_distance computes, for each point in the source point cloud, the distance to the target point cloud.",
    "open3d.camera.PinholeCameraIntrinsic stores the intrinsic camera matrix along with the image height and width.",
    "open3d.geometry.AxisAlignedBoundingBox is a class that defines an axis-aligned box that can be computed from 3D geometries. The bounding box is generated using the coordinate axes.",
    "open3d.t.geometry.PointCloud.remove_statistical_outliers removes points that are further away from their nb_neighbor neighbors in average. This function is not recommended to use on GPU.",
    "open3d.core.addmm performs the addmm operation on two 2D tensors with compatible shapes. Specifically, it returns output = alpha * A @ B + beta * input.",
    "pcl::ApproximateVoxelGrid::setLeafSize sets the voxel grid leaf size. It accepts the leaf size as an input parameter.",
    "pcl::StatisticalOutlierRemoval::setMeanK sets the number of nearest neighbors to use for mean distance estimation. The input parameter specifies the number of points to use for mean distance estimation.",
    "pcl::SACSegmentation::setMethodType sets the type of sample consensus method to use, as a user-given parameter. The method types are defined in method_types.h.",
    "pcl::PassThrough::setFilterFieldName provides the name of the field to be used for filtering data. In conjunction with setFilterLimits(), points having values outside the specified interval for this field will be discarded. The input parameter is the name of the field that will be used for filtering.",
    "pcl::gpu::EuclideanClusterExtraction::setClusterTolerance sets the spatial cluster tolerance as a measure in the L2 Euclidean space. The tolerance parameter defines the maximum distance between two points for them to be considered part of the same cluster.",
    "pcl::FPFHEstimation::computeFeature estimates the Fast Point Feature Histograms (FPFH) descriptors at a set of points given by setInputCloud() and setIndices(), using the surface in setSearchSurface() and the spatial locator in setSearchMethod(). The output is the point cloud model dataset containing the FPFH feature estimates.",
    "pcl::registration::TransformationEstimation estimates a rigid rotation transformation between a source and a target point cloud. It takes the source point cloud, a vector of indices describing the points of interest in the source cloud, and the target point cloud as inputs.",
    "pcl::RadiusOutlierRemoval::setMinNeighborsInRadius sets the number of neighbors that need to be present in order for a point to be classified as an inlier. The number of points within the radius set by setRadiusSearch() must be equal to or greater than this number. The default minimum is 1. The parameter is named min_pts.",
    "pcl::filters::Convolution3D::setRadiusSearch sets the sphere radius to be used for determining the nearest neighbors. The radius parameter specifies the maximum distance to consider a point a neighbor.",
    "pcl::gpu::EuclideanClusterExtraction::extract extracts clusters from a PointCloud given by setInputCloud() and setIndices(). The clusters parameter receives the resultant point clusters."
  ]  ,
    "answer": [],
    "contexts": []
,
}
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
api_key = os.environ.get("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=api_key
)

judge_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=api_key,
    model_kwargs={"response_format": {"type": "json_object"}}
)
metrics = [
    Faithfulness(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
    ContextPrecision(llm=judge_llm),
]

answers = []
contexts_list = []

questions = data["question"]
for question in questions:
    print(f"Question: {question}")
    answer, retrieved_docs = retrieve_answer(question)
    raw_contexts = [doc.page_content for doc in retrieved_docs]
    print(f"Answer: {answer}")
    print("End of answer.")
    answers.append(answer)
    contexts_list.append(raw_contexts)

data["answer"] = answers
data["contexts"] = contexts_list
dataset = Dataset.from_dict(data)

results = evaluate(dataset, metrics=metrics)