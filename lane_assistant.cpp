#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <iomanip>
#include <sys/timeb.h>

using namespace std;
class LaneDetector
{
public:
    double image_size;
    double image_center;
    bool left_flag = false;   // Tells us if there's left boundary of lane detected
    bool right_flag = false;  // Tells us if there's right boundary of lane detected
    cv::Point right_b;        // Members of both line equations of the lane boundaries:
    double right_m;           // y = m*x + b
    cv::Point left_b;         //
    double left_m;            //
    //bool lane_finder = 0;

    cv::Mat deNoise( cv::Mat inputImage )
    // Apply Gaussian blurring to the input Image
    {
        cv::Mat output;
        cv::GaussianBlur( inputImage, output, cv::Size( 3, 3 ), 0, 0 );
        return output;
    }

    std::vector<cv::Mat> edgeDetector( cv::Mat image_denoise )
    // Filter the image to obtain only edges
    {
        std::vector<cv::Mat> output_edge( 2 );
        cv::Mat output;
        cv::Mat output_yellow;
        cv::Mat output_white;
		
		cv::Mat kernel;
        cv::Point anchor;

		//yellow and white value in HSV image
		int yLowH = 11;
        int yHighH = 25;
        int yLowS = 43;
        int yHighS = 255;
        int yLowV = 46;
        int yHighV = 255;
        
	    /*int wLowH = 0;
        int wHighH = 180;
        int wLowS = 0;
        int wHighS = 30;
        int wLowV = 150;
        int wHighV = 255;*/

		cv::cvtColor( image_denoise, output, cv::COLOR_BGR2HSV );
        vector<cv::Mat> hsvsplit;
		split( output, hsvsplit );
        cv::equalizeHist( hsvsplit[2], hsvsplit[2] );
        merge( hsvsplit, output );
        
		// Use different ways because of better performance
		cv::cvtColor( image_denoise, output_white, cv::COLOR_RGB2GRAY );
		
        cv::threshold( output_white, output_white, 140, 255, cv::THRESH_BINARY );

		//inRange( output, cv::Scalar( wLowH, wLowS, wLowV ), cv::Scalar( wHighH, wHighS, wHighV ), output_white );
        inRange( output, cv::Scalar( yLowH, yLowS, yLowV ), cv::Scalar( yHighH, yHighS, yHighV), output_yellow );
		
		imshow( "w", output_white );
        imshow( "y", output_yellow );

		anchor = cv::Point( -1, -1 );
        kernel = cv::Mat( 1, 3, CV_32F );
        kernel.at<float>( 0, 0 ) = -1;
        kernel.at<float>( 0, 1 ) = 0;
        kernel.at<float>( 0, 2 ) = 1;

        cv::filter2D( output_yellow, output_yellow, -1, kernel, anchor, 0, cv::BORDER_DEFAULT );
        cv::filter2D( output_white, output_white, -1, kernel, anchor, 0, cv::BORDER_DEFAULT );

		output_edge[0] = output_yellow;
        output_edge[1] = output_white;



		return output_edge;
	}


    cv::Mat mask( cv::Mat image_edges )
    // Mask the edges image to only care about the Road
    {
        cv::Mat output;
        cv::Mat mask = cv::Mat::zeros( image_edges.size(), image_edges.type() );
        cv::Point pts[4] = {cv::Point( 0, image_edges.rows ),
                            cv::Point( image_edges.cols, image_edges.rows ),
                            // cv::Point( image_edges.cols, 0.7 * image_edges.rows ),
                            cv::Point( image_edges.cols, 0.5 * image_edges.rows ),
                            // cv::Point( 0.2 * image_edges.cols, 0.55 * image_edges.rows ),
                            cv::Point( 0, 0.5 * image_edges.rows )};

        cv::fillConvexPoly( mask, pts, 4, cv::Scalar( 255, 0, 0 ) );
        cv::bitwise_and( image_edges, mask, output );
        //cv::imshow( "mask", output );

        return output;
    }

    std::vector<cv::Vec4i> houghLines( cv::Mat image_mask )
    // Detect Hough lines in masked edges image
    {
        std::vector<cv::Vec4i> line;

        HoughLinesP( image_mask, line, 1, CV_PI / 180, 20, 20, 30 );

        return line;
    }

    std::vector<std::vector<cv::Vec4i>> lineSeparation( std::vector<cv::Vec4i> lines,
                                                        cv::Mat image_edges )
    // Sprt detected lines by their slope into right and left lines
    {
        std::vector<std::vector<cv::Vec4i>> output( 2 );
        size_t j = 0;
        cv::Point ini;
        cv::Point fini;
        double slope_thresh = 0.3;
        std::vector<double> slopes;
        std::vector<cv::Vec4i> selected_lines;
        std::vector<cv::Vec4i> right_lines, left_lines;
       

        for( auto i : lines )
        {
            ini = cv::Point( i[0], i[1] );
            fini = cv::Point( i[2], i[3] );

            double slope =
                ( static_cast<double>( fini.y ) - static_cast<double>( ini.y ) ) /
                ( static_cast<double>( fini.x ) - static_cast<double>( ini.x ) + 0.00001 );

            if( std::abs( slope ) > slope_thresh )
            {
                slopes.push_back( slope );
                selected_lines.push_back( i );
            }
        }

        image_center = static_cast<double>( ( image_edges.cols / 2 ) );
        while( j < selected_lines.size() )
        {
            ini = cv::Point( selected_lines[j][0], selected_lines[j][1] );
            fini = cv::Point( selected_lines[j][2], selected_lines[j][3] );

            if( slopes[j] > 0 && fini.x > image_center && ini.x > image_center )
            {
                right_lines.push_back( selected_lines[j] );
                right_flag = true;
            }
            else if( slopes[j] < 0 && fini.x < image_center && ini.x < image_center )
            {
                left_lines.push_back( selected_lines[j] );
                left_flag = true;
            }
            j++;
        }

        output[0] = right_lines;
        output[1] = left_lines;

        //if ((!output[0].empty()) && (!output[1].empty()))
		//{
         //       lane_finder = 1;
		//}

       return output;
    }

    std::vector<cv::Point> regression( std::vector<std::vector<cv::Vec4i>> left_right_lines,
                                       cv::Mat inputImage )
    {
        std::vector<cv::Point> output( 5 );
        cv::Point ini;
        cv::Point fini;
        cv::Point ini2;
        cv::Point fini2;
        cv::Vec4d right_line;
        cv::Vec4d left_line;
        std::vector<cv::Point> right_pts;
        std::vector<cv::Point> left_pts;

        if( right_flag == true )
        {
            for( auto i : left_right_lines[0] )
            {
                ini = cv::Point( i[0], i[1] );
                fini = cv::Point( i[2], i[3] );

                right_pts.push_back( ini );
                right_pts.push_back( fini );
            }

            if( right_pts.size() > 0 )
            {
                // The right line is formed here
                cv::fitLine( right_pts, right_line, cv::DIST_L2, 0, 0.01, 0.01 );
                right_m = right_line[1] / right_line[0];
                right_b = cv::Point( right_line[2], right_line[3] );  
            }
        }

        if( left_flag == true )
        {
            for( auto j : left_right_lines[1] )
            {
                ini2 = cv::Point( j[0], j[1] );
                fini2 = cv::Point( j[2], j[3] );

                left_pts.push_back( ini2 );
                left_pts.push_back( fini2 );
            }

            if( left_pts.size() > 0 )
            {
                // The left line is formed here
                cv::fitLine( left_pts, left_line, cv::DIST_L2, 0, 0.01, 0.01 );
                left_m = left_line[1] / left_line[0];
                left_b = cv::Point( left_line[2], left_line[3] );
            }
        }

        int ini_y = inputImage.rows;
        int fin_y = 0.6 * inputImage.rows;
        // int mark_y = 0.8 * inputImage.rows;

        double right_ini_x = ( ( ini_y - right_b.y ) / right_m ) + right_b.x;
        double right_fin_x = ( ( fin_y - right_b.y ) / right_m ) + right_b.x;
        // double right_mark_x = ( (mark_y - right_b.y) / right_m ) + right_b.x;

        double left_ini_x = ( ( ini_y - left_b.y ) / left_m ) + left_b.x;
        double left_fin_x = ( ( fin_y - left_b.y ) / left_m ) + left_b.x;
        // double left_mark_x = ( ( mark_y - left_b.y ) / left_m ) + left_b.x;

        double mark_center_x = 0.5 * ( right_fin_x + left_fin_x );
		//get the center of the street

        output[0] = cv::Point( right_ini_x, ini_y );
        output[1] = cv::Point( right_fin_x, fin_y );
        output[2] = cv::Point( left_ini_x, ini_y );
        output[3] = cv::Point( left_fin_x, fin_y );
        // output[4] = cv::Point( right_mark_x, mark_y );
        // output[5] = cv::Point( left_mark_x, mark_y );
        output[4] = cv::Point( mark_center_x, 0 );

        return output;
    }

    std::string predictTurn()
    {
        std::string output;
        double vanish_x;
        double thr_vp = 10;

        // The vanishing point is the point where both lane boundary lines intersect
        vanish_x = static_cast<double>(
            ( ( right_m * right_b.x ) - ( left_m * left_b.x ) - right_b.y + left_b.y ) /
            ( right_m - left_m ) );

        // The vanishing points location determines where is the road turning
        if( vanish_x < ( image_center - thr_vp ) )
            output = "Left Turn";
        else if( vanish_x > ( image_center + thr_vp ) )
            output = "Right Turn";
        else if( vanish_x >= ( image_center - thr_vp ) && vanish_x <= ( image_center + thr_vp ) )
            output = "Straight";

        return output;
    }

    int plotLane( cv::Mat inputImage, std::vector<cv::Point> lane,
                  std::string turn )  // Plot the resultant lane and turn prediction in the frame.

    {
        std::vector<cv::Point> poly_points;
        cv::Mat output;

        // Create the transparent polygon for a better visualization of the lane
        inputImage.copyTo( output );
        poly_points.push_back( lane[2] );
        poly_points.push_back( lane[0] );
        poly_points.push_back( lane[1] );
        poly_points.push_back( lane[3] );
        cv::fillConvexPoly( output, poly_points, cv::Scalar( 250, 206, 135 ), cv::LINE_AA, 0 );
        cv::addWeighted( output, 0.3, inputImage, 1.0 - 0.3, 0, inputImage );

        cv::putText( inputImage, turn, cv::Point( 50, 90 ), cv::FONT_HERSHEY_COMPLEX_SMALL, 3,
                     cv::Scalar( 0, 255, 0 ), 1, cv::LINE_AA );

        // Plot both lines of the lane boundary
        cv::line( inputImage, lane[0], lane[1], cv::Scalar( 0, 255, 0 ), 5, cv::LINE_AA );
        cv::line( inputImage, lane[2], lane[3], cv::Scalar( 0, 255, 0 ), 5, cv::LINE_AA );
        // cv::line( inputImage, lane[4], lane[5], cv::Scalar( 0, 255, 0 ), 5,
        // cv::LINE_AA );

        return 0;
    }
};

class PID
{
public:
    double p_error_;
    double i_error_;
    double d_error_;

    double Kp_;
    double Ki_;
    double Kd_;

    PID()
        : p_error_( 0 ),
          i_error_( 0 ),
          d_error_( 0 ),
          Kp_( 0 ),
          Ki_( 0 ),
          Kd_( 0 ),
          has_prev_cte_( 0 ),
          prev_cte_( 0 )
    {
    }

    void Init( double Kp, double Ki, double Kd )
    {
        Kp_ = Kp;
        Ki_ = Ki;
        Kd_ = Kd;

        p_error_ = 0;
        i_error_ = 0;
        d_error_ = 0;

        prev_cte_ = 0;
        has_prev_cte_ = false;
    }

    void UpdateError( double cte )
    {
        const int threshold = 1000;//maximum to clean the value i_error_

        if( i_error_ > threshold )
        {
            i_error_ = 0;
        }

        if( !has_prev_cte_ )
        {
            //std::cout << "ok in if" << std::endl;
            prev_cte_ = cte;
            has_prev_cte_ = true;
        }

        /*std::cout << "cte:" << setprecision( 6 ) << cte << std::endl;
        std::cout << "prev1:" << setprecision( 6 ) << prev_cte_ << std::endl;*/
        p_error_ = cte;
        i_error_ = i_error_ + cte;
        d_error_ = cte - prev_cte_;

        /*std::cout << "p:" << setprecision( 6 ) << cte << std::endl;
        std::cout << "d:" << setprecision( 6 ) << d_error_ << std::endl;*/

        prev_cte_ = cte;
        //std::cout << "prev2:" << setprecision( 6 ) << prev_cte_ << std::endl;
    }

    double TotalError()
    {
        return -Kp_ * p_error_ - Ki_ * i_error_ - Kd_ * d_error_;
    }

    bool has_prev_cte_;
    double prev_cte_;
};

class LaneAssistant
{
    // insert your custom functions and algorithms here
public:
    LaneAssistant()
    {
		//initialize two PID controller
        pid_steering.Init( -0.0035, 0, -0.0055 );
        pid_throttle.Init( -0.1, -0.004, -5 );
    }

    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
		// do stuff with data
        // send results via socket
		
		/*get the frequency 
        timeb t;
        ftime( &t );
        std::cout << t.time * 1000 + t.millitm << endl;*/
        
		//control the frequency of PID = 0.05s
		static int count = 0;
        count++;
        if( count != 10 )
            return true;

		//set the maximum velocity
        double max_velocity = 60;
        double throttle_value = 0;
        // string throttlevalue = "throttle_value:";

        double cte = 0;//steering error
        double steer_value;
        // string steervalue = "steering_value:";

		//set the distance we should keep according to ISO15622-2018
        const double time_gap = 2.3;
        double dist_soll = 0.01 * time_gap * ego_velocity_;  // cm/s * s *0.01 = m
        
		double dist_ist;//the distance we have

        //if lanes have been detected, use PID control steering
		if( !lane_1.empty() )
        {
            if( std::abs( lane_1[4].x - lanedetector.image_center ) > 0 )
            {
                cte = lane_1[4].x - lanedetector.image_center;
                //std::cout << "4:" << cte << std::endl;
            }

            pid_steering.UpdateError( cte );
            steer_value = pid_steering.TotalError();

            //if only one lane is detected, give a slight steering value
			if( !lanedetector.left_flag && lanedetector.right_flag )
            {
                steer_value = -0.02;
                cte = 0;
            }
            else if( lanedetector.left_flag && !lanedetector.right_flag )
            {
                steer_value = 0.02;
                cte = 0;
            }
            // std::cout << lane_1[4].x << std::endl;
            // std::cout << lanedetector.image_center << std::endl;
            // std::cout << cte << std::endl;
            //std::cout << "steer:" << steer_value << std::endl;

            // socket.send( tronis::SocketData( steervalue + to_string( steer_value ) ) );
        }

		//whenever the ego-velocity exceeds the max, exit ACC
        if( ( max_velocity >= 0.036 * ego_velocity_ ) )
        {
            if( !Object_dect.empty() )
            {
				//find the distance to the nearest object
                dist_ist = *min_element( Object_dect.begin(), Object_dect.end() );

                double diff_dist = dist_ist - dist_soll;

                /*std::cout << "soll:" << setprecision( 6 ) << dist_soll << std::endl;

                std::cout << "min:" << setprecision( 6 ) << dist_ist << std::endl;

                std::cout << "diff:" << setprecision( 6 ) << diff_dist << std::endl;*/

                pid_throttle.UpdateError( diff_dist );
                throttle_value = pid_throttle.TotalError();

                //nomalize the throttle value to 0-1
				if( (throttle_value > -4) && (diff_dist > -0.2 * dist_soll) && (dist_ist > 5))
                {
                    throttle_value = 0.07 * throttle_value + 0.58;  //max(6,1)min(-4,0.3)
                }
                else
                {
					//if we are too close behind the lead vehicle, stop
                    throttle_value = 0;  
                }
                
				//std::cout << "throttle:" << throttle_value << std::endl;
            }
            else
            {
				//no obstacle, accelerate
                throttle_value = 0.8;
                //std::cout << "throttle1:" << throttle_value << std::endl;
            }
        }
        else
        {
			//whenever exceeds the max velocity, brake
            throttle_value = 0;
        }
        
		socket.send(
            tronis::SocketData( to_string( throttle_value ) + ":" + to_string( steer_value ) ) );

        count = 0;
        return true;
    }

protected:
    LaneDetector lanedetector;
    PID pid_steering;
    PID pid_throttle;
    std::string image_name_;
    cv::Mat image_;
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;
    std::vector<cv::Point> lane_1;
    std::vector<double> Object_dect; // collect objects which need to be avoided
  
    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        // do stuff
        cv::Mat image_denoise;
        cv::Mat image_mask_y;
        cv::Mat image_mask_w;
        cv::Mat image_lines;
		
		std::vector<cv::Mat> image_edges;
        std::vector<cv::Vec4i> lines_y;
        std::vector<cv::Vec4i> lines_w;

        std::vector<std::vector<cv::Vec4i>> left_right_lines;
        std::vector<cv::Point> lane;
        std::string turn;
        
		int flag_plot = -1;
		bool yellow_finder = 0;

        image_denoise = lanedetector.deNoise( image_ );

        image_edges = lanedetector.edgeDetector( image_denoise );
		
		//cv::imshow( "edges", image_edges );

        image_mask_y = lanedetector.mask( image_edges[0] );
        image_mask_w = lanedetector.mask( image_edges[1] );

		cv::imshow( "image_mask_y", image_mask_y );
        cv::imshow( "image_mask_w", image_mask_w );

        lines_y = lanedetector.houghLines( image_mask_y );
        lines_w = lanedetector.houghLines( image_mask_w );

        if( (!lines_y.empty())||(!lines_w.empty()) )
        {
            //find the lines in yellow
			left_right_lines = lanedetector.lineSeparation( lines_y, image_edges[0] );
            
			/*for( std::vector<cv::Vec4i>::const_iterator i = left_right_lines[0].begin();
                 i != left_right_lines[0].end(); ++i )
            {
                            std::cout << " yellow" << ( *i )[0] << ( *i )[1] << ( *i )[2]
                                      << ( *i )[3] << ' ';
            }

				for( std::vector<cv::Vec4i>::const_iterator i = left_right_lines[1].begin();
                 i != left_right_lines[1].end(); ++i )
            {
                std::cout << " yellow" << ( *i )[0] << ( *i )[1] << ( *i )[2] << ( *i )[3] << ' ';
            }*/

			yellow_finder =
                ( ( !left_right_lines[0].empty() ) && ( !left_right_lines[1].empty() ) );

			//std::cout << "Line" << yellow_finder << std::endl;

			if(yellow_finder == 1)//if there are yellow lanes, driving along them
			{
				lane = lanedetector.regression( left_right_lines, image_ );

                turn = lanedetector.predictTurn();

                flag_plot = lanedetector.plotLane( image_, lane, turn );
			}
            else//if no yellow lanes, along white lanes
			{
				left_right_lines = lanedetector.lineSeparation( lines_w, image_edges[1] );
                
				lane = lanedetector.regression( left_right_lines, image_ );

                turn = lanedetector.predictTurn();

                flag_plot = lanedetector.plotLane( image_, lane, turn );
			}
		}
        else
        {
            flag_plot = -1;
        }

        lane_1 = lane;
    }

    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = msg->Velocity;

        //std::cout << "velocity:" << 0.036 * ego_velocity_ << endl;
        return true;
    }

    bool processObject( tronis::BoxDataSub* msg )
    {
        // do stuff
        std::cout << "ok in function" << endl;
        tronis::BoxDataSub* sensorData = msg;

        double dist_;
        //std::cout << "1.0" << std::endl;

		//clean the previous vector
        Object_dect.swap( vector<double>() );

        for( size_t i = 0; i < sensorData->Objects.size(); i++ )
        {
            tronis::ObjectSub& object = sensorData->Objects[i];
            std::cout << object.ActorName.Value() << " at ";
            std::cout << object.Pose.Location.ToString() << std::endl;
            // std::cout << object.BB.Origin.ToString() << std::endl;

            tronis::LocationSub location = object.Pose.Location;
            std::string ID = object.ActorName.Value();
            bool elim_ = ( ( ID != "TronisTrack_1" ) && ( ID != "CameraActor_0" ) &&
                           ( ID != "TronisTrack_6" ) && ( ID != "TronisTrack_4" ) );

            float pos_x = location.X;
            float pos_y = location.Y;
            float pos_z = location.Z;


            //std::cout << "type is" << int( object.Type.Value() ) << std::endl;

            dist_ = 0.01 * sqrt( pow( pos_x, 2 ) + pow( pos_y, 2 ) );  // m
            //std::cout << "distance:" << dist_ << std::endl;

            //filter out the moveable object we should avoid
			if( ( object.Type.Value() == 1 ) && ( std::abs( pos_y ) < 440 ) && ( dist_ < 50 ) &&
                elim_ )
            {
                Object_dect.push_back( dist_ );
                std::cout << " get object!" << std::endl;
            }
            else
            {
                std::cout << " no object!" << std::endl;
            }
        }

        /*for (std::vector<double>::const_iterator i = Object_dect.begin(); i!= Object_dect.end();
  ++i)
    {
  std::cout << " diatance_str" << *i << ' ';
        }      */

        return true;
    }

    // Helper functions, no changes needed
public:
    // Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
                      << ", Time: " << data_model->GetTime() << std::endl;
            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
                    processImage( data_model->GetName(),
                                  data_model.get_typed<tronis::ImageSub>()->Image );
                    break;
                }
                case tronis::TronisDataType::ImageFrame:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFrameSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::ImageFramePose:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::PoseVelocity:
                {
                    processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                    break;
                }
                case tronis::TronisDataType::BoxData:
                {
                    processObject( data_model.get_typed<tronis::BoxDataSub>() );
                    break;
                }
                default:
                {
                    std::cout << data_model->ToString() << std::endl;
                    break;
                }
            }
            return true;
        }
        else
        {
            std::cout << data_model->ToString() << std::endl;
            return false;
        }
    }

protected:
    // Function to show an openCV image in a separate window
    void showImage( std::string image_name, cv::Mat image )
    {
        cv::Mat out = image;
        if( image.type() == CV_32F || image.type() == CV_64F )
        {
            cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
        }
        cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );
        cv::imshow( image_name.c_str(), out );
    }

    // Function to convert tronis image to openCV image
    bool processImage( const std::string& base_name, const tronis::Image& image )
    {
        std::cout << "processImage" << std::endl;
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_ = tronis::image2Mat( image );

        detectLanes();
        showImage( image_name_, image_ );

        return true;
    }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    std::cout << "Welcome to lane assistant" << std::endl;

    // specify socket parameters
    std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
                  << "\", PortBind:" << socket_port << "}";

    int key_press = 0;  // close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param

    LaneAssistant lane_assistant;

    while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
        tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
            // wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
                // data received! reset timer
                time_ms = 0;

                // convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
                // identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
                // no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
