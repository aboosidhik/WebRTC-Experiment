/*! \file    posst-processing.c
 * \author   Aboobeker Sidhik <aboosidhik@gmail.com>
 * \copyright GNU General Public License v3
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>  

extern "C"
{
#include <arpa/inet.h>
#ifdef __MACH__
#include <machine/endian.h>
#else
#include <endian.h>
#endif
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <glib.h>
#include <jansson.h>
#include <vpx/vpx_decoder.h>
#include <vpx/vp8dx.h>
}
using namespace std;
using namespace cv;
string type2str(int type) { 
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

#define fourcc 0x30395056
#define interface (&vpx_codec_vp9_dx_algo)


static int kVp9FrameMarker = 2;
static int kMinTileWidthB64 = 4;
static int kMaxTileWidthB64 = 64;
static int kRefFrames = 8;
static int kRefsPerFrame = 3;
static int kRefFrames_LOG2 = 3;
static int kVpxCsBt601 = 1;
static int kVpxCsSrgb = 7;
static int kVpxCrStudioRange = 0;
static int kVpxCrFullRange = 1;
static int kMiSizeLog2 = 3;
static int bit_depth_ = 0;
static  int profile_ = -1;
static  int show_existing_frame_ = 0;
static  int key_ = 0;
static  int altref_ = 0;
static  int error_resilient_mode_ = 0;
static  int intra_only_ = 0;
static  int reset_frame_context_ = 0;
  
static  int color_space_ = 0;
static  int color_range_ = 0;
static  int subsampling_x_ = 0;
static  int subsampling_y_ = 0;
static  int refresh_frame_flags_ = 0;
static  int width_;
static  int height_;
static  int row_tiles_;
static  int column_tiles_;
static  int frame_parallel_mode_;
static int fm_count;
#define htonll(x) ((1==htonl(1)) ? (x) : ((gint64)htonl((x) & 0xFFFFFFFF) << 32) | htonl((x) >> 32))
#define ntohll(x) ((1==ntohl(1)) ? (x) : ((gint64)ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((x) >> 32))

typedef struct janus_pp_rtp_header
{
#if __BYTE_ORDER == __BIG_ENDIAN
	uint16_t version:2;
	uint16_t padding:1;
	uint16_t extension:1;
	uint16_t csrccount:4;
	uint16_t markerbit:1;
	uint16_t type:7;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
	uint16_t csrccount:4;
	uint16_t extension:1;
	uint16_t padding:1;
	uint16_t version:2;
	uint16_t type:7;
	uint16_t markerbit:1;
#endif
	uint16_t seq_number;
	uint32_t timestamp;
	uint32_t ssrc;
	uint32_t csrc[16];
} janus_pp_rtp_header;

typedef struct janus_pp_rtp_header_extension {
	uint16_t type;
	uint16_t length;
} janus_pp_rtp_header_extension;

typedef struct janus_pp_frame_packet {
        uint16_t seq;	/* RTP Sequence number */
	uint64_t ts;	/* RTP Timestamp */
	uint16_t len;	/* Length of the data */
	int pt;			/* Payload type of the data */
	long offset;	/* Offset of the data in the file */
	int skip;		/* Bytes to skip, besides the RTP header */
	uint8_t drop;	/* Whether this packet can be dropped (e.g., padding)*/
	struct janus_pp_frame_packet *next;
	struct janus_pp_frame_packet *prev;
} janus_pp_frame_packet;

typedef struct frame_packet {
        AVFrame *frame;
        AVPacket *pkt;
        struct frame_packet *next;
	struct frame_packet *prev;
} frame_packet;

typedef struct file_av {
    char *source;
    FILE *file;
    long fsize;
    long offset;
    int opus ;
    int vp9;
    int count;
    gboolean parsed_header;
    janus_pp_frame_packet *list;
    janus_pp_frame_packet *last;
    gint64 c_time;
    gint64 w_time;
    uint32_t last_ts;
    uint32_t reset;
    AVCodecContext *codec_ctx;
    AVCodec *codec_dec; 
    int times_resetted;
    int numBytes;
    uint8_t *received_frame;
    uint8_t *buffer;
    uint8_t *start;
    int max_width, max_height, fps;
    int min_ts_diff, max_ts_diff;
    uint32_t post_reset_pkts;
    int len, frameLen;
    int audio_len;
    int keyFrame;
    uint32_t keyframe_ts;
    int64_t audio_ts;
    int audio_pts;
    int video_pts;
    int audio;
    int video;
    gchar *buf;
    struct file_av *next;
    struct file_av *prev;
}file_av;

typedef struct file_av_list {
    size_t size;
    struct file_av *head;
    struct file_av *tail;
}file_av_list;

typedef struct file_combine {
        int num;
        char *audio_source;
        char *video_source;
        file_av_list *file_av_list_1;
	struct file_combine *next;
	struct file_combine *prev;
} file_combine;

typedef struct file_combine_list {
    size_t size;
    struct file_combine *head;
    struct file_combine *tail;
}file_combine_list;



int janus_log_level = 4;
gboolean janus_log_timestamps = FALSE;
gboolean janus_log_colors = TRUE;

int working = 0;


/* Signal handler */
void janus_pp_handle_signal(int signum);
void janus_pp_handle_signal(int signum) {
	working = 0;
}
/*! \file    pp-webm.c
 * \author   Lorenzo Miniero <lorenzo@meetecho.com>
 * \copyright GNU General Public License v3
 * \brief    Post-processing to generate .webm files
 * \details  Implementation of the post-processing code (based on FFmpeg)
 * needed to generate .webm files out of VP8/VP9 RTP frames.
 *
 * \ingroup postprocessing
 * \ref postprocessing
 */


/* WebRTC stuff (VP8/VP9) */
#if defined(__ppc__) || defined(__ppc64__)
	# define swap2(d)  \
	((d&0x000000ff)<<8) |  \
	((d&0x0000ff00)>>8)
#else
	# define swap2(d) d
#endif

#define LIBAVCODEC_VER_AT_LEAST(major, minor) \
	(LIBAVCODEC_VERSION_MAJOR > major || \
	 (LIBAVCODEC_VERSION_MAJOR == major && \
	  LIBAVCODEC_VERSION_MINOR >= minor))

#if LIBAVCODEC_VER_AT_LEAST(51, 42)
#define PIX_FMT_YUV420P AV_PIX_FMT_YUV420P
#endif


/* WebM output */
static AVFormatContext *fctx;
static AVStream *vStream;
static AVStream *aStream;
static int max_width = 0, max_height = 0, fps = 0;
static AVRational audio_timebase;
static AVRational video_timebase;
static AVOutputFormat *fmt;
static AVCodec *audio_codec;
static AVCodec *video_codec;
static AVDictionary *opt_arg;
static AVCodecContext *context;
static AVCodecContext *video_context;
int janus_pp_webm_create(char *destination) {
	if(destination == NULL)
		return -1;
#if LIBAVCODEC_VERSION_MAJOR < 55
	printf("Your FFmpeg version does not support VP9\n");
	return -1;
	
#endif
	/* Setup FFmpeg */
	av_register_all();
        avformat_alloc_output_context2(&fctx, NULL, NULL, destination);
        if (!fctx) {
            printf("Could not deduce output format from file extension: using WEBM.\n");
            avformat_alloc_output_context2(&fctx, fmt, "webm", destination);
        }
        if (!fctx) {
            return -1;
        }    
        fmt = fctx->oformat;
        audio_codec = avcodec_find_encoder(AV_CODEC_ID_OPUS);
        video_codec = avcodec_find_encoder(AV_CODEC_ID_VP9);
        vStream = avformat_new_stream(fctx, NULL);
        aStream = avformat_new_stream(fctx, NULL);
        if (!aStream) {
            printf("Could not allocate audio stream\n");
            return -1;
        } 
        if (!vStream) {
            printf("Could not allocate video stream\n");
            return -1;
        }
        vStream->id = fctx->nb_streams-1;
        aStream->id = fctx->nb_streams-1;
        video_context = avcodec_alloc_context3(video_codec);
        context = avcodec_alloc_context3(audio_codec);
        if (!context) {
            printf("Could not alloc an encoding context\n");
            return -1;
        } 
        if (!video_context) {
            printf("Could not alloc an encoding context\n");
            return -1;
        }
        context->codec_type = AVMEDIA_TYPE_AUDIO;
        context->codec_id = AV_CODEC_ID_OPUS;
        context->sample_fmt = AV_SAMPLE_FMT_S16;
        context->bit_rate = 64000;
        context->sample_rate = 48000;
        context->channel_layout = AV_CH_LAYOUT_STEREO;
        context->channels = 2;
        aStream->time_base = (AVRational){ 1, context->sample_rate};
        audio_timebase = (AVRational){ 1, context->sample_rate};
        video_context->codec_type = AVMEDIA_TYPE_VIDEO;
        video_context->codec_id = AV_CODEC_ID_VP9;
        video_context->width = 1200;
	video_context->height = 900;
        vStream->time_base = (AVRational){1, 30};
        video_timebase = (AVRational){1, 30};
        video_context->time_base =vStream->time_base;
        video_context->pix_fmt = AV_PIX_FMT_YUV420P;
       /* Some formats want stream headers to be separate. */
        if (fctx->oformat->flags & AVFMT_GLOBALHEADER) {
            context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            video_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }    
        int ret;
        /* open it */
       
        opt_arg = NULL;
        ret = avcodec_open2(context, audio_codec, &opt_arg);
        if (ret < 0) {
            printf("Could not open audio codec\n");
            return -1;
        }
        /* 
        ret = avcodec_open2(node->video_context, node->video_codec, &node->opt_arg);
         if (ret < 0) {
            fprintf(stderr, "Could not open video codec: %s\n", av_err2str(ret));
            return NULL;
        }
       */      
        /* copy the stream parameters to the muxer */
        ret = avcodec_parameters_from_context(vStream->codecpar, video_context);
        if (ret < 0) {
            printf("Could not copy the stream parameters\n");
            return -1;
        }        
        /* copy the stream parameters to the muxer */
        ret = avcodec_parameters_from_context(aStream->codecpar, context);
        if (ret < 0) {
            printf("Could not copy the stream parameters\n");
            return -1;
        }
        av_dump_format(fctx, 0, destination, 1);
            /* open the output file, if needed */
        if (!(fmt->flags & AVFMT_NOFILE)) {
            ret = avio_open(&fctx->pb, destination, AVIO_FLAG_WRITE);
            if (ret < 0) {
                printf("Could not open file\n");
                return -1;
            }
        }
        /* Write the stream header, if any. */
        ret = avformat_write_header(fctx, &opt_arg);
        if (ret < 0) {
                printf("Error occurred when opening output file\n");
                return -1;
        }
        /*
	fctx = avformat_alloc_context();
	if(fctx == NULL) {
		printf( "Error allocating context\n");
		return -1;
	}
	fctx->oformat = av_guess_format("webm", NULL, NULL);
	if(fctx->oformat == NULL) {
		printf( "Error guessing format\n");
		return -1;
	}
	snprintf(fctx->filename, sizeof(fctx->filename), "%s", destination);
	//~ vStream = av_new_stream(fctx, 0);
	vStream = avformat_new_stream(fctx, 0);
	if(vStream == NULL) {
		printf("Error adding stream\n");
		return -1;
	}
	//~ avcodec_get_context_defaults2(vStream->codec, CODEC_TYPE_VIDEO);
#if LIBAVCODEC_VER_AT_LEAST(53, 21)
	avcodec_get_context_defaults3(vStream->codec, AVMEDIA_TYPE_VIDEO);
#else
	avcodec_get_context_defaults2(vStream->codec, AVMEDIA_TYPE_VIDEO);
#endif
#if LIBAVCODEC_VER_AT_LEAST(54, 25)
	#if LIBAVCODEC_VERSION_MAJOR >= 55
	vStream->codec->codec_id = vp8 ? AV_CODEC_ID_VP8 : AV_CODEC_ID_VP9;
	#else
	vStream->codec->codec_id = AV_CODEC_ID_VP8;
	#endif
#else
	vStream->codec->codec_id = CODEC_ID_VP8;
#endif
	//~ vStream->codec->codec_type = CODEC_TYPE_VIDEO;
	vStream->codec->codec_type = AVMEDIA_TYPE_VIDEO;
	vStream->codec->time_base = (AVRational){1, fps};
	vStream->codec->width = max_width;
	vStream->codec->height = max_height;
	vStream->codec->pix_fmt = PIX_FMT_YUV420P;
	if (fctx->flags & AVFMT_GLOBALHEADER)
		vStream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
	//~ fctx->timestamp = 0;
	//~ if(url_fopen(&fctx->pb, fctx->filename, URL_WRONLY) < 0) {
	if(avio_open(&fctx->pb, fctx->filename, AVIO_FLAG_WRITE) < 0) {
		printf( "Error opening file for output\n");
		return -1;
	}
	//~ memset(&parameters, 0, sizeof(AVFormatParameters));
	//~ av_set_parameters(fctx, &parameters);
	//~ fctx->preload = (int)(0.5 * AV_TIME_BASE);
	//~ fctx->max_delay = (int)(0.7 * AV_TIME_BASE);
	//~ if(av_write_header(fctx) < 0) {
	if(avformat_write_header(fctx, NULL) < 0) {
		printf( "Error writing header\n");
		return -1;
	}
        
        */
	return 0;
}

int janus_pp_webm_preprocess(file_av *file_av_1) {
	if(!file_av_1->file || !file_av_1->list)
		return -1;
	janus_pp_frame_packet *tmp = file_av_1->list;
	int bytes = 0; 
        file_av_1->min_ts_diff = 0;
        file_av_1->max_ts_diff = 0;
	char prebuffer[1500];
	memset(prebuffer, 0, 1500);
       
	while(tmp) {
		if(tmp == file_av_1->list || tmp->ts > tmp->prev->ts) {
			if(tmp->prev != NULL && tmp->ts > tmp->prev->ts) {
				int diff = tmp->ts - tmp->prev->ts;
				if(file_av_1->min_ts_diff == 0 || file_av_1->min_ts_diff > diff)
					file_av_1->min_ts_diff = diff;
				if(file_av_1->max_ts_diff == 0 || file_av_1->max_ts_diff < diff)
					file_av_1->max_ts_diff = diff;
			}
			if(tmp->prev != NULL && (tmp->seq - tmp->prev->seq > 1)) {
				printf("Lost a packet here? (got seq %"SCNu16" after %"SCNu16", time ~%"SCNu64"s)\n",
					tmp->seq, tmp->prev->seq, (tmp->ts-file_av_1->list->ts)/90000);
			}
		}
		if(tmp->drop) {
			/* We marked this packet as one to drop, before */
			printf("Dropping previously marked video packet (time ~%"SCNu64"s)\n", (tmp->ts-file_av_1->list->ts)/90000);
			tmp = tmp->next;
			continue;
		}
		/* https://tools.ietf.org/html/draft-ietf-payload-vp9 */
		/* Read the first bytes of the payload, and get the first octet (VP9 Payload Descriptor) */
		fseek(file_av_1->file, tmp->offset+12+tmp->skip, SEEK_SET);
		bytes = fread(prebuffer, sizeof(char), 16, file_av_1->file);
		if(bytes != 16)
                    printf("Didn't manage to read all the bytes we needed (%d < 16)...\n", bytes);
		char *buffer = (char *)&prebuffer;
		uint8_t vp9pd = *buffer;
		uint8_t ibit = (vp9pd & 0x80);
		uint8_t pbit = (vp9pd & 0x40);
		uint8_t lbit = (vp9pd & 0x20);
		uint8_t fbit = (vp9pd & 0x10);
		uint8_t vbit = (vp9pd & 0x02);
		//printf("%" PRIu8 ",%" PRIu8 ",%" PRIu8 ",%" PRIu8 ",%" PRIu8 "\n", ibit,pbit,lbit,fbit,vbit);
		buffer++;
		if(ibit) {
                    /* Read the PictureID octet */
                    vp9pd = *buffer;
                    uint16_t picid = vp9pd, wholepicid = picid;
                    uint8_t mbit = (vp9pd & 0x80);
                    if(!mbit) {
                    	buffer++;
                    } else {
					memcpy(&picid, buffer, sizeof(uint16_t));
					wholepicid = ntohs(picid);
					picid = (wholepicid & 0x7FFF);
					buffer += 2;
				}
			}
			if(lbit) {
				buffer++;
				if(!fbit) {
					/* Non-flexible mode, skip TL0PICIDX */
					buffer++;
				}
			}
			if(fbit && pbit) {
				/* Skip reference indices */
				uint8_t nbit = 1;
				while(nbit) {
					vp9pd = *buffer;
					nbit = (vp9pd & 0x01);
					buffer++;
				}
			}
			if(vbit) {
				/* Parse and skip SS */
				vp9pd = *buffer;
				uint n_s = (vp9pd & 0xE0) >> 5;
				n_s++;
				uint8_t ybit = (vp9pd & 0x10);
				if(ybit) {
					/* Iterate on all spatial layers and get resolution */
					buffer++;
					uint i=0;
					for(i=0; i<n_s; i++) {
						/* Width */
						uint16_t *w = (uint16_t *)buffer;
						int width = ntohs(*w);
						buffer += 2;
						/* Height */
						uint16_t *h = (uint16_t *)buffer;
						int height = ntohs(*h);
						buffer += 2;
						if(width > file_av_1->max_width)
							file_av_1->max_width = width;
						if(height > file_av_1->max_height)
							file_av_1->max_height = height;
					}
				}
			}
		
		tmp = tmp->next;
	}

	int mean_ts = file_av_1->min_ts_diff;	/* FIXME: was an actual mean, (max_ts_diff+min_ts_diff)/2; */
	file_av_1->fps = (90000/(mean_ts > 0 ? mean_ts : 30));
	printf( "  -- %dx%d (fps [%d,%d] ~ %d)\n", file_av_1->max_width, file_av_1->max_height, file_av_1->min_ts_diff, file_av_1->max_ts_diff, file_av_1->fps);
	if(file_av_1->max_width == 0 && file_av_1->max_height == 0) {
		printf("No key frame?? assuming 640x480...\n");
		file_av_1->max_width = 640;
		file_av_1->max_height = 480;
	}
	if(file_av_1->fps == 0) {
		printf("No fps?? assuming 1...\n");
		file_av_1->fps = 1;	/* Prevent divide by zero error */
	}
	return 0;
}

int ReadBit(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  const size_t off = bit_offset_;
  const size_t byte_offset = off >> 3;
  const int bit_shift = 7 - (int)(off & 0x7);
  if (byte_offset < frame_size_) {
    const int bit = (frame_[byte_offset] >> bit_shift) & 1;
    bit_offset_++;
    return bit;
  } else {
    return 0;
  }
}
int VpxReadLiteral(int bits, const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  int value = 0;
  int bit;
  for (bit = bits - 1; bit >= 0; --bit)
    value |= ReadBit(frame_,frame_size_,bit_offset_) << bit;
  return value;
}

int ValidateVp9SyncCode(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  const int sync_code_0 = VpxReadLiteral(8,frame_,frame_size_,bit_offset_);
  const int sync_code_1 = VpxReadLiteral(8,frame_,frame_size_,bit_offset_);
  const int sync_code_2 = VpxReadLiteral(8,frame_,frame_size_,bit_offset_);
  return (sync_code_0 == 0x49 && sync_code_1 == 0x83 && sync_code_2 == 0x42);
}

void ParseColorSpace(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  bit_depth_ = 0;
  if (profile_ >= 2)
    bit_depth_ = ReadBit(frame_,frame_size_,bit_offset_) ? 12 : 10;
  else
    bit_depth_ = 8;
  color_space_ = VpxReadLiteral(3,frame_,frame_size_,bit_offset_);
  if (color_space_ != kVpxCsSrgb) {
    color_range_ = ReadBit(frame_,frame_size_,bit_offset_);
    if (profile_ == 1 || profile_ == 3) {
      subsampling_x_ = ReadBit(frame_,frame_size_,bit_offset_);
      subsampling_y_ = ReadBit(frame_,frame_size_,bit_offset_);
      ReadBit(frame_,frame_size_,bit_offset_);
    } else {
      subsampling_y_ = subsampling_x_ = 1;
    }
  } else {
    color_range_ = kVpxCrFullRange;
    if (profile_ == 1 || profile_ == 3) {
      subsampling_y_ = subsampling_x_ = 0;
      ReadBit(frame_,frame_size_,bit_offset_);
    }
  }
}

void ParseFrameResolution(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  width_ = VpxReadLiteral(16,frame_,frame_size_,bit_offset_) + 1;
  height_ = VpxReadLiteral(16,frame_,frame_size_,bit_offset_) + 1;
   printf("width:%d, height:%d\n", width_,height_);
}

void ParseFrameParallelMode(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  if (ReadBit(frame_,frame_size_,bit_offset_)) {
    VpxReadLiteral(16,frame_,frame_size_,bit_offset_);  // display width
    VpxReadLiteral(16,frame_,frame_size_,bit_offset_);  // display height
  }
  if (!error_resilient_mode_) {
    ReadBit(frame_,frame_size_,bit_offset_);  // Consume refresh frame context
    frame_parallel_mode_ = ReadBit(frame_,frame_size_,bit_offset_);
  } else {
    frame_parallel_mode_ = 1;
  }
}

void SkipDeltaQ(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  if (ReadBit(frame_,frame_size_,bit_offset_))
    VpxReadLiteral(4,frame_,frame_size_,bit_offset_);
}

int AlignPowerOfTwo(int value, int n) {
  return (((value) + ((1 << (n)) - 1)) & ~((1 << (n)) - 1));
}

void ParseTileInfo(const uint8_t* frame_, size_t frame_size_, size_t bit_offset_) {
  VpxReadLiteral(2,frame_,frame_size_,bit_offset_);  // Consume frame context index

  // loopfilter
  VpxReadLiteral(6,frame_,frame_size_,bit_offset_);  // Consume filter level
  VpxReadLiteral(3,frame_,frame_size_,bit_offset_);  // Consume sharpness level

  int mode_ref_delta_enabled = ReadBit(frame_,frame_size_,bit_offset_);
  if (mode_ref_delta_enabled) {
    int mode_ref_delta_update = ReadBit(frame_,frame_size_,bit_offset_);
    if (mode_ref_delta_update) {
      const int kMaxRefLFDeltas = 4;
      int i;
      for (i = 0; i < kMaxRefLFDeltas; ++i) {
        if (ReadBit(frame_,frame_size_,bit_offset_))
          VpxReadLiteral(7,frame_,frame_size_,bit_offset_);  // Consume ref_deltas + sign
      }

      const int kMaxModeDeltas = 2;
      for (i = 0; i < kMaxModeDeltas; ++i) {
        if (ReadBit(frame_,frame_size_,bit_offset_))
          VpxReadLiteral(7,frame_,frame_size_,bit_offset_);  // Consume mode_delta + sign
      }
    }
  }
    // quantization
  VpxReadLiteral(8,frame_,frame_size_,bit_offset_);  // Consume base_q
  SkipDeltaQ(frame_,frame_size_,bit_offset_);  // y dc
  SkipDeltaQ(frame_,frame_size_,bit_offset_);  // uv ac
  SkipDeltaQ(frame_,frame_size_,bit_offset_);  // uv dc

  // segmentation
  int segmentation_enabled = ReadBit(frame_,frame_size_,bit_offset_);
  if (!segmentation_enabled) {
    const int aligned_width = AlignPowerOfTwo(width_, kMiSizeLog2);
    const int mi_cols = aligned_width >> kMiSizeLog2;
    const int aligned_mi_cols = AlignPowerOfTwo(mi_cols, kMiSizeLog2);
    const int sb_cols = aligned_mi_cols >> 3;  // to_sbs(mi_cols);
    int min_log2_n_tiles, max_log2_n_tiles;

    for (max_log2_n_tiles = 0;
         (sb_cols >> max_log2_n_tiles) >= kMinTileWidthB64;
         max_log2_n_tiles++) {
    }
    max_log2_n_tiles--;
    if (max_log2_n_tiles < 0)
      max_log2_n_tiles = 0;

    for (min_log2_n_tiles = 0; (kMaxTileWidthB64 << min_log2_n_tiles) < sb_cols;
         min_log2_n_tiles++) {
    }

    // columns
    const int max_log2_tile_cols = max_log2_n_tiles;
    const int min_log2_tile_cols = min_log2_n_tiles;
    int max_ones = max_log2_tile_cols - min_log2_tile_cols;
    int log2_tile_cols = min_log2_tile_cols;
    while (max_ones-- && ReadBit(frame_,frame_size_,bit_offset_))
      log2_tile_cols++;

    // rows
    int log2_tile_rows = ReadBit(frame_,frame_size_,bit_offset_);
    if (log2_tile_rows)
      log2_tile_rows += ReadBit(frame_,frame_size_,bit_offset_);

    row_tiles_ = 1 << log2_tile_rows;
    column_tiles_ = 1 << log2_tile_cols;
  }
}

int ParseUncompressedHeader(const uint8_t* frame, size_t length) {
  if (!frame || length == 0)
    return 0;
  const uint8_t* frame_ = frame;
  size_t frame_size_ = length;
  size_t bit_offset_ = 0;
  int bits = 2;
  int value = 0;
  int bit;

  for (bit = bits - 1; bit >= 0; --bit)
    value |= ReadBit(frame_, frame_size_,bit_offset_) << bit;
  const int frame_marker = value;
   printf("marker:%d\n",value);
   profile_ = ReadBit(frame_, frame_size_,bit_offset_);
  profile_ |= ReadBit(frame_, frame_size_,bit_offset_) << 1;
  if (profile_ > 2)
    profile_ += ReadBit(frame_, frame_size_,bit_offset_);  
  
  
  show_existing_frame_ = ReadBit(frame_, frame_size_,bit_offset_);
  key_ = !ReadBit(frame_, frame_size_,bit_offset_);
  altref_ = !ReadBit(frame_, frame_size_,bit_offset_);
  error_resilient_mode_ = ReadBit(frame_, frame_size_,bit_offset_);
  if (key_) {
    if (!ValidateVp9SyncCode(frame_, frame_size_,bit_offset_)) {
      printf("Invalid Sync code!\n");
      return 0;
    }
   //printf("frame_marker:%d, profile:%d\n", frame_marker,profile_);
    ParseColorSpace(frame_,frame_size_,bit_offset_);
    ParseFrameResolution(frame_,frame_size_,bit_offset_);
    ParseFrameParallelMode(frame_,frame_size_,bit_offset_);
    ParseTileInfo(frame_,frame_size_,bit_offset_);
  } else {
    intra_only_ = altref_ ? ReadBit(frame_,frame_size_,bit_offset_) : 0;
    reset_frame_context_ = error_resilient_mode_ ? 0 : VpxReadLiteral(2,frame_,frame_size_,bit_offset_);
    if (intra_only_) {
      if (!ValidateVp9SyncCode(frame_,frame_size_,bit_offset_)) {
        printf("Invalid Sync code!\n");
        return 0;
      }

      if (profile_ > 0) {
        ParseColorSpace(frame_,frame_size_,bit_offset_);
      } else {
        // NOTE: The intra-only frame header does not include the specification
        // of either the color format or color sub-sampling in profile 0. VP9
        // specifies that the default color format should be YUV 4:2:0 in this
        // case (normative).
        color_space_ = kVpxCsBt601;
        color_range_ = kVpxCrStudioRange;
        subsampling_y_ = subsampling_x_ = 1;
        bit_depth_ = 8;
      }

      refresh_frame_flags_ = VpxReadLiteral(kRefFrames,frame_,frame_size_,bit_offset_);
      ParseFrameResolution(frame_,frame_size_,bit_offset_);
    } else {
      refresh_frame_flags_ = VpxReadLiteral(kRefFrames,frame_,frame_size_,bit_offset_);
      int i;
      for ( i = 0; i < kRefsPerFrame; ++i) {
        VpxReadLiteral(kRefFrames_LOG2,frame_,frame_size_,bit_offset_);  // Consume ref.
        ReadBit(frame_,frame_size_,bit_offset_);  // Consume ref sign bias.
      }
      //printf(" else frame_marker:%d, profile:%d\n", frame_marker,profile_);
      int found = 0;
      for ( i = 0; i < kRefsPerFrame; ++i) {
        if (ReadBit(frame_,frame_size_,bit_offset_)) {
          // Found previous reference, width and height did not change since
          // last frame.
          found = 1;
          break;
        }
      }

      if (!found)
        ParseFrameResolution(frame_,frame_size_,bit_offset_);
    }
  }

  
  return 1;
}



int janus_pp_webm_process(file_combine_list *file_combine_list_1, int *working) {
	if(!file_combine_list_1)
		return -1;
        FILE *f;
        f = fopen("ss.h264", "wb");
        //video decoding of all :
        // 1) parse vp9 header and send to decoder for all files
        //mux using cuda
        // encode using h264 hardware 
        //mux audio and video into mp4 file
        cv::VideoWriter out("output1.h264", CV_FOURCC('x','2', '6', '4'), 32, cv::Size(400,300));
        if(!out.isOpened()) {
            cout <<"Error! Unable to open video file for output." << std::endl;
            exit(-1);
        }
        AVFormatContext *fctx_v;
        AVStream *vStream_v;
        int max_width_v = 0, max_height_v = 0, fps_v = 0;
        AVRational video_timebase_v;
        AVOutputFormat *fmt_v;
        AVCodec *video_codec_v;
        AVDictionary *opt_arg_v;
        AVCodecContext *video_context_v;
        char *destination = (char *)malloc(sizeof(char)*128);
        destination = "ott.mp4";
        /* Setup FFmpeg */
	av_register_all();
        avformat_alloc_output_context2(&fctx_v, NULL, NULL, destination);
        if (!fctx_v) {
            printf("Could not deduce output format from file extension: using WEBM.\n");
            avformat_alloc_output_context2(&fctx_v, fmt, "mp4", destination);
        }
        if (!fctx_v) {
            return -1;
        }    
        fmt_v = fctx_v->oformat;
        video_codec_v = avcodec_find_encoder(AV_CODEC_ID_H264);
        vStream_v = avformat_new_stream(fctx_v, NULL);
        if (!vStream_v) {
            printf("Could not allocate video stream\n");
            return -1;
        }
        vStream_v->id = fctx_v->nb_streams-1;
        video_context_v = avcodec_alloc_context3(video_codec_v);
        if (!video_context_v) {
            printf("Could not alloc an encoding context\n");
            return -1;
        }
        video_context_v->codec_type = AVMEDIA_TYPE_VIDEO;
        video_context_v->codec_id = AV_CODEC_ID_H264;
        video_context_v->width = 400;
	video_context_v->height = 300;
        vStream_v->time_base = (AVRational){1, 32};
        video_timebase_v = (AVRational){1, 32};
        video_context_v->time_base = vStream_v->time_base;
        video_context_v->pix_fmt = AV_PIX_FMT_YUV420P;
        video_context_v->gop_size = 10;
        video_context_v->max_b_frames = 1;
       /* Some formats want stream headers to be separate. */
        if (fctx_v->oformat->flags & AVFMT_GLOBALHEADER) {
            video_context_v->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }  
        
        int ret;
        opt_arg_v = NULL;
        /* 
        ret = avcodec_open2(node->video_context, node->video_codec, &node->opt_arg);
         if (ret < 0) {
            fprintf(stderr, "Could not open video codec: %s\n", av_err2str(ret));
            return NULL;
        }
       */      
        /* copy the stream parameters to the muxer */
        ret = avcodec_parameters_from_context(vStream_v->codecpar, video_context_v);
        if (ret < 0) {
            printf("Could not copy the stream parameters\n");
            return -1;
        }        
        /* copy the stream parameters to the muxer */
        av_dump_format(fctx_v, 0, destination, 1);
            /* open the output file, if needed */
        if (!(fmt_v->flags & AVFMT_NOFILE)) {
            ret = avio_open(&fctx_v->pb, destination, AVIO_FLAG_WRITE);
            if (ret < 0) {
                printf("Could not open file\n");
                return -1;
            }
        }
        /* Write the stream header, if any. */
        ret = avformat_write_header(fctx_v, &opt_arg_v);
        if (ret < 0) {
                printf("Error occurred when opening output file\n");
                return -1;
        }
        
        
        // video decoding
        file_combine *file_combine_1 = file_combine_list_1->head;
        int j,m;
         int frame_cnt = 0;
        for(j = 0; j <file_combine_list_1->size; j++) {
            file_av *file_av_1 = file_combine_1->file_av_list_1->head;
            for(m = 0; m < file_combine_1->file_av_list_1->size; m++) {
                if(file_av_1->vp9 == 1) {
                    avcodec_register_all();
                    file_av_1->codec_dec = avcodec_find_decoder_by_name("libvpx-vp9");
                    AVCodec *codec_enc = avcodec_find_encoder(AV_CODEC_ID_H264);
                    file_av_1->codec_ctx = avcodec_alloc_context3(file_av_1->codec_dec);
                    AVCodecContext *codec_enc_ctx = avcodec_alloc_context3(codec_enc);
                    codec_enc_ctx->width = 400;
                    codec_enc_ctx->height = 300;
                    codec_enc_ctx->time_base = (AVRational){1, 32};
                    codec_enc_ctx->framerate = (AVRational){32,1};
                    codec_enc_ctx->gop_size = 10;
                    codec_enc_ctx->max_b_frames = 1;
                    codec_enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
                    //av_opt_set(codec_enc_ctx->priv_data, "preset", "slow", 0);

                    if (avcodec_open2(codec_enc_ctx, codec_enc, NULL) < 0)
                        exit(1);
                    if (avcodec_open2(file_av_1->codec_ctx, file_av_1->codec_dec, NULL) < 0)
                        exit(1);
                    AVPacket *pkt = av_packet_alloc();
                    if (!pkt)
                        exit(1);
                    AVPacket *pkt_enc = av_packet_alloc();
                    if (!pkt_enc)
                        exit(1);
                    AVFrame *frame = av_frame_alloc();
                    if (!frame) {
                        fprintf(stderr, "Could not allocate video frame\n");
                        exit(1);
                    }
                    file_av_1->codec_ctx->width = 400;
                    file_av_1->codec_ctx->height = 300;
                    file_av_1->codec_ctx->framerate = (AVRational){32,1};
                    janus_pp_frame_packet *tmp = file_av_1->list;
                    file_av_1->numBytes = file_av_1->max_width*file_av_1->max_height*3;
                    file_av_1->received_frame = (uint8_t*)g_malloc0(file_av_1->numBytes);
                    file_av_1->buffer = (uint8_t*)g_malloc0(10000);
                    file_av_1->start = file_av_1->buffer;
                    uint8_t *buffer = (uint8_t*)g_malloc0(10000), *start = buffer;
                    file_av_1->len = 0, file_av_1->frameLen = 0;
                    file_av_1->audio_len = 0;
                    file_av_1->keyFrame = 0;
                    file_av_1->keyframe_ts = 0;
                    file_av_1->audio_ts = 0;
                    file_av_1->audio_pts = 0;
                    file_av_1->video_pts = 0;
                    file_av_1->audio = 0;
                    file_av_1->video = 0;
                    int bytes = 0;
                    while(*working && tmp != NULL) {        
                        file_av_1->keyFrame = 0;
                        file_av_1->frameLen = 0;
                        file_av_1->len = 0;
                        file_av_1->audio_len = 0;
                        file_av_1->buf = (gchar*)g_malloc0(1000);
                        while(1) {
                            fm_count++;
                            if(tmp->drop) {
                                // Check if timestamp changes: marker bit is not mandatory, and may be lost as well 
                                if(tmp->next == NULL || tmp->next->ts > tmp->ts)
                                    break;
                                tmp = tmp->next;
                                continue;
                            }
                            // RTP payload 
                            buffer = start;
                            file_av_1->buffer = file_av_1->start;
                            fseek(file_av_1->file, tmp->offset+12+tmp->skip, SEEK_SET);
                            file_av_1->len = tmp->len-12-tmp->skip;
                            bytes = fread(buffer, sizeof(char), file_av_1->len, file_av_1->file);
                            if(bytes != file_av_1->len)
                                printf("Didn't manage to read all the bytes we needed (%d < %d)...\n", bytes, file_av_1->len);
                            
				int skipped = 0;
				uint8_t vp9pd = *buffer;
				uint8_t ibit = (vp9pd & 0x80);
				uint8_t pbit = (vp9pd & 0x40);
				uint8_t lbit = (vp9pd & 0x20);
				uint8_t fbit = (vp9pd & 0x10);
				uint8_t vbit = (vp9pd & 0x02);
				/* Move to the next octet and see what's there */
				buffer++;
				file_av_1->len--;
				skipped++;
				if(ibit) {
					/* Read the PictureID octet */
					vp9pd = *buffer;
					uint16_t picid = vp9pd, wholepicid = picid;
					uint8_t mbit = (vp9pd & 0x80);
					if(!mbit) {
						buffer++;
						file_av_1->len--;
						skipped++;
					} else {
						memcpy(&picid, buffer, sizeof(uint16_t));
						wholepicid = ntohs(picid);
						picid = (wholepicid & 0x7FFF);
						buffer += 2;
						file_av_1->len -= 2;
						skipped += 2;
					}
				}
				if(lbit) {
					buffer++;
					file_av_1->len--;
					skipped++;
					if(!fbit) {
						/* Non-flexible mode, skip TL0PICIDX */
						buffer++;
						file_av_1->len--;
						skipped++;
					}
				}
				if(fbit && pbit) {
					/* Skip reference indices */
					uint8_t nbit = 1;
					while(nbit) {
						vp9pd = *buffer;
						nbit = (vp9pd & 0x01);
						buffer++;
						file_av_1->len--;
						skipped++;
					}
				}
				if(vbit) {
					/* Parse and skip SS */
					vp9pd = *buffer;
					int n_s = (vp9pd & 0xE0) >> 5;
					n_s++;
					uint8_t ybit = (vp9pd & 0x10);
					uint8_t gbit = (vp9pd & 0x08);
					if(ybit) {
						/* Iterate on all spatial layers and get resolution */
						buffer++;
						file_av_1->len--;
						skipped++;
						int i=0;
						for(i=0; i<n_s; i++) {
							/* Been there, done that: skip skip skip */
							buffer += 4;
							file_av_1->len -= 4;
							skipped += 4;
						}
						/* Is this the first keyframe we find?
						 * (FIXME assuming this really means "keyframe...) */
						if(file_av_1->keyframe_ts == 0) {
							file_av_1->keyframe_ts = tmp->ts;
							printf( "First keyframe: %"SCNu64"\n", tmp->ts-file_av_1->list->ts);
						}
					}
					if(gbit) {
						if(!ybit) {
							buffer++;
							file_av_1->len--;
							skipped++;
						}
						uint8_t n_g = *buffer;
						buffer++;
						file_av_1->len--;
						skipped++;
						if(n_g > 0) {
							int i=0;
							for(i=0; i<n_g; i++) {
								/* Read the R bits */
								vp9pd = *buffer;
								int r = (vp9pd & 0x0C) >> 2;
								if(r > 0) {
									/* Skip reference indices */
									buffer += r;
									file_av_1->len -= r;
									skipped += r;
								}
								buffer++;
								file_av_1->len--;
								skipped++;
							}
						}
					}
				}
                            memcpy(file_av_1->received_frame + file_av_1->frameLen, buffer, file_av_1->len);
                            file_av_1->frameLen += file_av_1->len;
                            if(file_av_1->len == 0)
                                break;
                            if(tmp->next == NULL || tmp->next->ts > tmp->ts)
                                break;
                            tmp = tmp->next;
                        }
                        int rr = 0;
                        tmp = tmp->next;
                        pkt->data = file_av_1->received_frame;
                        pkt->size = file_av_1->frameLen;
                        if(file_av_1->keyFrame)
                            //~ packet.flags |= PKT_FLAG_KEY;
                            pkt->flags |= AV_PKT_FLAG_KEY;
                        
                        pkt->dts = (tmp->ts-file_av_1->keyframe_ts)/90;
                        pkt->pts = (tmp->ts-file_av_1->keyframe_ts)/90;                        
                        ret = avcodec_send_packet(file_av_1->codec_ctx, pkt);
                        if (ret < 0) {
                            printf("Error sending a packet for decoding\n");
                            continue;
                        }
                        ret = avcodec_receive_frame(file_av_1->codec_ctx, frame);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                                continue;
                        else if (ret < 0) {
                                printf("Error during decoding\n");
                                continue;
                            } 
                        printf("%" PRId64 ", %" PRId64 "\n", frame->pkt_pts, frame->pkt_dts);
                    /*  uint32_t pitchY = frame->linesize[0];
                        uint32_t pitchU = frame->linesize[1];
                        uint32_t pitchV = frame->linesize[2];

                        uint8_t *avY = frame->data[0];
                        uint8_t *avU = frame->data[1];
                        uint8_t *avV = frame->data[2];

                        for (uint32_t i = 0; i < frame->height; i++) {
                            fwrite(avY, frame->width, 1, f);
                            avY += pitchY;
                        }

                        for (uint32_t i = 0; i < frame->height/2; i++) {
                            fwrite(avU, frame->width/2, 1, f);
                            avU += pitchU;
                        }

                        for (uint32_t i = 0; i < frame->height/2; i++) {
                            fwrite(avV, frame->width/2, 1, f);
                            avV += pitchV;
                        }
                      */  
                        
                      /*  ret = avcodec_send_frame(codec_enc_ctx,frame);
                        if (ret < 0) {
                            fprintf(stderr, "Error sending a frame for encoding\n");
                            continue;
                        } else if(ret == 0) {
                            printf("sucess encoding\n");
                        }
                        while (ret >= 0) {
                            ret = avcodec_receive_packet(codec_enc_ctx, pkt_enc);
                            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF){
                                printf("what the fuck \n");
                                continue;
                            } else if (ret < 0) {
                                fprintf(stderr, "Error during encoding\n");
                                continue;
                            }
                            if(fctx_v) {
                                if(av_write_frame(fctx_v, pkt_enc) < 0) {
                                    printf("Error writing video frame to file...\n");
                                }
                            }   
                           //printf("Write frame %3"PRId64" (size=%5d)\n", pkt_enc->pts, pkt_enc->size);
                            //fwrite(pkt_enc->data, 1, pkt_enc->size, f);
                            //av_packet_unref(pkt_enc);
                        } */
                        //printf("bb%" PRId64 ", bb %" PRId64 "\n", pkt_enc->dts, pkt_enc->pts);
                    
                       /*  
                        pkt_enc->pts = pkt_enc->dts;
                        if(ret == 0)
                            printf("sucess receiving\n");    
                        if(fctx_v) {
                            if(av_write_frame(fctx_v, pkt_enc) < 0) {
                                printf("Error writing video frame to file...\n");
                            }
                        }
			*/            
                        struct SwsContext *convert_ctx;
                        Mat m;
                        AVFrame dst;
                        int w = frame->width;
                        int h = frame->height;
                        m = cv::Mat(h, w, CV_8UC3);
                        dst.data[0] = (uint8_t *)m.data;
                        avpicture_fill( (AVPicture *)&dst, dst.data[0], AV_PIX_FMT_BGR24, w, h);

                        enum AVPixelFormat src_pixfmt = (enum AVPixelFormat)frame->format;
                        enum AVPixelFormat dst_pixfmt = AV_PIX_FMT_BGR24;
                        convert_ctx = sws_getContext(w, h, src_pixfmt, w, h, dst_pixfmt, SWS_FAST_BILINEAR, NULL, NULL, NULL);

                        if(convert_ctx == NULL) {
                            fprintf(stderr, "Cannot initialize the conversion context!\n");
                            continue;
                        }

                        sws_scale(convert_ctx, frame->data, frame->linesize, 0, h,
                                            dst.data, dst.linesize);
                        Mat image_re(m.rows,m.cols, CV_8UC3);
                        cuda::GpuMat d_src(m);
                        cuda::GpuMat d_src_image(image_re);
                        cuda::GpuMat d_dst1;
                        cuda::GpuMat d_dst2;
                        cuda::GpuMat d_dst_big;
                        cuda::resize(d_src, d_dst1, Size(m.cols/2,m.rows/2), 0, 0, INTER_CUBIC);  
                        cuda::resize(d_src, d_dst2, Size(m.cols/2,m.rows/2), 0, 0, INTER_CUBIC);  
                        cuda::resize(d_src, d_dst_big, Size(m.cols/2,m.rows), 0, 0, INTER_CUBIC);  
                        d_dst1.copyTo(d_src_image(cv::Rect(200,0,d_dst1.cols, d_dst1.rows)));
                        d_dst_big.copyTo(d_src_image(cv::Rect(0,0,d_dst_big.cols, d_dst_big.rows)));
                        d_dst2.copyTo(d_src_image(cv::Rect(200,150,d_dst2.cols, d_dst2.rows)));
                        d_src_image.download(image_re);
                        out << image_re;
                        
                        //out <<m;
                    /*    file_av_1->len = 0;
                        if(tmp_audio->drop) {
                            // Check if timestamp changes: marker bit is not mandatory, and may be lost as well 
                            if(tmp_audio->next == NULL || tmp_audio->next->ts > tmp_audio->ts)
                                break;
                            tmp_audio = tmp_audio->next;
                            continue;
                        }
                        fseek(file_audio, tmp_audio->offset+12+tmp_audio->skip, SEEK_SET);
                        audio_len = tmp_audio->len-12-tmp_audio->skip;
                        bytes = fread(buf, sizeof(char), audio_len, file_audio);
                    //}
                    
                    //from here audio +video
                    video_pts = (tmp->ts-keyframe_ts)/90;
                    if(video_pts == 0 && audio_pts == 0) {
                        memset(received_frame + frameLen, 0, FF_INPUT_BUFFER_PADDING_SIZE);
                        AVPacket packet;
                        av_init_packet(&packet); 
                        packet.stream_index = 0;
                        packet.data = received_frame;
                        packet.size = frameLen;
                        if(keyFrame)
                            //~ packet.flags |= PKT_FLAG_KEY;
                            packet.flags |= AV_PKT_FLAG_KEY;
                        // First we save to the file... 
                        //~ packet.dts = AV_NOPTS_VALUE;
                        //~ packet.pts = AV_NOPTS_VALUE;
                        // printf("Error writing  to audio file 1 .. of user %" PRId64 "...\n", audio_ts);
                        packet.dts = (tmp->ts-keyframe_ts)/90;
                        packet.pts = (tmp->ts-keyframe_ts)/90;
			if(fctx) {
                            if(av_write_frame(fctx, &packet) < 0) {
                                printf("Error writing video frame to file...\n");
                            }
                        }
                        AVPacket packet1;
                        av_init_packet(&packet1); 
                        packet1.dts = audio_pts;
                        packet1.pts = audio_pts;
                        packet1.data = buf;
                        packet1.size = audio_len;
                        packet1.stream_index = 1;
                        if(fctx) {
                            if(av_write_frame(fctx, &packet1) < 0) {
                                //printf("Error writing  to audio file 1 .. of user %lu...\n", node->id);
                                tmp_audio = tmp_audio->next;
                                audio_ts = audio_ts + 20;
                                g_free(buf);
                                //continue;
                            }    
                        }
                        audio_pts = audio_pts + 20;
                        printf("Audio and video %" PRId64 " %" PRId64 "...\n", audio_pts, video_pts);
                        audio = 1;
                        video = 1;
                        tmp = tmp->next;
                        tmp_audio = tmp_audio->next;
                    } else {
                        AVPacket packet;
                        av_init_packet(&packet); 
                        if(video_pts < audio_pts) {
                            memset(received_frame + frameLen, 0, FF_INPUT_BUFFER_PADDING_SIZE);
                            packet.stream_index = 0;
                            packet.data = received_frame;
                            packet.size = frameLen;
                            if(keyFrame)
                                //~ packet.flags |= PKT_FLAG_KEY;
                                packet.flags |= AV_PKT_FLAG_KEY;

                            // First we save to the file... 
                            //~ packet.dts = AV_NOPTS_VALUE;
                            //~ packet.pts = AV_NOPTS_VALUE;
                            // printf("Error writing  to audio file 1 .. of user %" PRId64 "...\n", audio_ts);
                            packet.dts = (tmp->ts-keyframe_ts)/90;
                            packet.pts = (tmp->ts-keyframe_ts)/90;
                            printf("video only %" PRId64 "  %"PRId64 "...\n", video_pts, audio_pts);
                            if(fctx) {
                                if(av_write_frame(fctx, &packet) < 0) {
                                    printf("Error writing video frame to file...\n");
                                }
                            }
                            tmp = tmp->next;
                            audio = 1;
                            video = 0;
                        } else if (video_pts > audio_pts){
                            packet.dts = audio_pts;
                            packet.pts = audio_pts;
                            packet.data = buf;
                            packet.size = audio_len;
                            printf("Audio only %" PRId64 " %" PRId64 "...\n", audio_pts, video_pts);
                            packet.stream_index = 1;
                            if(fctx) {
                                if(av_write_frame(fctx, &packet) < 0) {
                                    //printf("Error writing  to audio file 1 .. of user %lu...\n", node->id);
                                }    
                            }
                            audio_pts = audio_pts + 20;
                            tmp_audio= tmp_audio->next;
                            video = 1;
                            audio = 0;
                        } else if (video_pts == audio_pts) {
                            memset(received_frame + frameLen, 0, FF_INPUT_BUFFER_PADDING_SIZE);
                            AVPacket packet;
                            av_init_packet(&packet); 
                            packet.stream_index = 0;
                            packet.data = received_frame;
                            packet.size = frameLen;
                            if(keyFrame)
				//~ packet.flags |= PKT_FLAG_KEY;
				packet.flags |= AV_PKT_FLAG_KEY;

                            // First we save to the file... 
                            //~ packet.dts = AV_NOPTS_VALUE;
                            //~ packet.pts = AV_NOPTS_VALUE;
                            // printf("Error writing  to audio file 1 .. of user %" PRId64 "...\n", audio_ts);
                            packet.dts = (tmp->ts-keyframe_ts)/90;
                            packet.pts = (tmp->ts-keyframe_ts)/90;
                            if(fctx) {
                                if(av_write_frame(fctx, &packet) < 0) {
                                    printf("Error writing video frame to file...\n");
                                }
                            }
                            AVPacket packet1;
                            av_init_packet(&packet1); 
                            packet1.dts = audio_pts;
                            packet1.pts = audio_pts;
                            packet1.data = buf;
                            packet1.size = audio_len;
                            packet1.stream_index = 1;
                            if(fctx) {
                                if(av_write_frame(fctx, &packet1) < 0) {
                                    //printf("Error writing  to audio file 1 .. of user %lu...\n", node->id);
                                    tmp_audio = tmp_audio->next;
                                    audio_ts = audio_ts + 20;
                                    g_free(buf);
                                    //continue;
                                }    
                            }
                            audio_pts = audio_pts + 20;
                            printf("Audio and video %" PRId64 " %" PRId64 "...\n", audio_pts, video_pts);
                            audio = 1;
                            video = 1;
                            tmp = tmp->next;
                            tmp_audio = tmp_audio->next;
                        }
                    }
                    */
                    }
                    g_free(file_av_1->received_frame);
                    g_free(file_av_1->start);   
                }
                file_av_1 = file_av_1->next;
            }
            file_combine_1 = file_combine_1->next;
        }    
        return 0;
}


/* Close WebM file */
void janus_pp_webm_close(void) {
	if(fctx != NULL)
		av_write_trailer(fctx);
	if(vStream != NULL && vStream->codec != NULL)
		avcodec_close(vStream->codec);
	if(fctx != NULL && fctx->streams[0] != NULL) {
		av_free(fctx->streams[0]->codec);
		av_free(fctx->streams[0]);
	}
	if(fctx != NULL) {
		//~ url_fclose(fctx->pb);
		avio_close(fctx->pb);
		av_free(fctx);
	}
}

/* Main Code */
int main(int argc, char *argv[])
{
        fm_count = 0;
        int i, j, m;
        char *destination = (char *)malloc(sizeof(char)*128);
        char *extension = (char *)malloc(sizeof(char)*128);
        
        printf("Enter the number of files for audio and video mixing: \n");
        scanf("%d",&i);
        printf("Enter the destination file: \n");
        scanf("%s",destination);
        extension = strrchr(destination, '.');
	if(extension == NULL) {
            /* No extension? */
            printf( "No extension? Unsupported target file\n");
            exit(1);
        }
	if(strcasecmp(extension, ".webm")) {
            /* Unsupported extension? */
            printf( "Unsupported extension '%s'\n", extension);
            exit(1);
        }
        file_combine_list *file_combine_list_1 = (file_combine_list *)malloc(sizeof(file_combine_list_1));
        file_combine_list_1->size = 0;
        file_combine_list_1->head = NULL;
        file_combine_list_1->tail = NULL;   
        for (j = 0; j<i; j++) {
            file_combine *number_source = (file_combine*) malloc(sizeof(file_combine));
            number_source->file_av_list_1 =  (file_av_list*) malloc(sizeof(file_av_list));
            number_source->file_av_list_1->size = 0;
            number_source->file_av_list_1->head = NULL;
            number_source->file_av_list_1->tail = NULL;   
            int m;
            for(m = 0;  m<2; m++) {
                int p;
                file_av *file_av_1 = (file_av*) malloc(sizeof(file_av));
                printf("Enter 1 for audio files and 2 for video files \n");
                scanf("%d",&p);
                file_av_1->source =  (char*)malloc(sizeof(char)*128);
                if( p == 1) {
                    printf("Enter audio file \n");
                    scanf("%s",file_av_1->source);
                    file_av_1->opus = 1;
                    file_av_1->vp9 = 0;
                } else if( p == 2) {
                    printf("Enter video file \n");
                    scanf("%s",file_av_1->source);
                    file_av_1->opus = 0;
                    file_av_1->vp9 = 1;
                }
                
                file_av_1->file = fopen(file_av_1->source, "rb");
                if(file_av_1->file == NULL) {
                    printf("Could not open one of the file \n");
                    return -1;
                }
                fseek(file_av_1->file, 0L, SEEK_END);
                file_av_1->fsize = ftell(file_av_1->file);
                fseek(file_av_1->file, 0L, SEEK_SET);
                printf("File is %zu bytes\n", file_av_1->fsize);
                if (number_source->file_av_list_1->head) {
                    // Binding the node to the list elements.
                    file_av_1->next = number_source->file_av_list_1->head;
                    file_av_1->prev = number_source->file_av_list_1->head->prev;
                    // Binding the list elements to the node.
                    number_source->file_av_list_1->head->prev->next = file_av_1;
                    number_source->file_av_list_1->head->prev = file_av_1;
                } else {
                    file_av_1->next = file_av_1;
                    file_av_1->prev = file_av_1;
                    number_source->file_av_list_1->tail = file_av_1;
                }
                number_source->file_av_list_1->head = file_av_1;
                number_source->file_av_list_1->size++;
            }
            if (file_combine_list_1->head) {
                // Binding the node to the list elements.
                number_source->next = file_combine_list_1->head;
                number_source->prev = file_combine_list_1->head->prev;
                // Binding the list elements to the node.
                file_combine_list_1->head->prev->next = number_source;
                file_combine_list_1->head->prev = number_source;
            } else {
                number_source->next = number_source;
                number_source->prev = number_source;
                file_combine_list_1->tail = number_source;
            }
            file_combine_list_1->head = number_source;
            file_combine_list_1->size++;
        }
	/* Handle SIGINT */
	working = 1;
	signal(SIGINT, janus_pp_handle_signal);


	/* Let's look for timestamp resets first */
        file_combine *file_combine_1 = file_combine_list_1->head;
        for (j = 0; j < i; j++) {
            file_av *file_av_1 = file_combine_1->file_av_list_1->head;
            for(m = 0; m < 2; m++ ) {
                working = 1;
                file_av_1->offset = 0;
                int bytes = 0, skip = 0;
                uint16_t len = 0;
                char prebuffer[1500];
                memset(prebuffer, 0, 1500);
                //printf("%i %i \n",file_av_1->opus,file_av_1->vp9);
                file_av_1->parsed_header = FALSE;
                while(working && file_av_1->offset < file_av_1->fsize) {
                        /* Read frame header */
                        skip = 0;
                        fseek(file_av_1->file, file_av_1->offset, SEEK_SET);
                        bytes = fread(prebuffer, sizeof(char), 8, file_av_1->file);
                        if(bytes != 8 || prebuffer[0] != 'M') {
                                printf("Invalid header at offset %ld (%s), the processing will stop here...\n",
                                        file_av_1->offset, bytes != 8 ? "not enough bytes" : "wrong prefix");
                                break;
                        }
                        if(prebuffer[1] == 'E') {
                                /* Either the old .mjr format header ('MEETECHO' header followed by 'audio' or 'video'), or a frame */
                                //printf("eee %i %i\n",file_av_1->offset, file_av_1->fsize);
                                file_av_1->offset += 8;
                                bytes = fread(&len, sizeof(uint16_t), 1, file_av_1->file);
                                len = ntohs(len);
                                file_av_1->offset += 2;
                                if(len < 12) {
                                    /* Not RTP, skip */
                                    printf("Skipping packet (not RTP?)\n");
                                    file_av_1->offset += len;
                                    continue;
                                }
                        } else if(prebuffer[1] == 'J') {
                                /* New .mjr format, the header may contain useful info */
                                file_av_1->offset += 8;
                                bytes = fread(&len, sizeof(uint16_t), 1, file_av_1->file);
                                len = ntohs(len);
                                file_av_1->offset += 2;
                                if(len > 0  && !file_av_1->parsed_header) {
                                        /* This is the info header */
                                        printf("New .mjr header format\n");
                                        bytes = fread(prebuffer, sizeof(char), len, file_av_1->file);
                                        file_av_1->parsed_header = TRUE;
                                        prebuffer[len] = '\0';
                                        json_error_t error;
                                        json_t *info = json_loads(prebuffer, 0, &error);
                                        if(!info) {
                                                printf("JSON error: on line %d: %s\n", error.line, error.text);
                                                printf("Error parsing info header...\n");
                                                exit(1);
                                        }
                                        /* Is it audio or video? */
                                        json_t *type = json_object_get(info, "t");
                                        if(!type || !json_is_string(type)) {
                                                printf("Missing/invalid recording type in info header...\n");
                                                exit(1);
                                        }
                                        const char *t = json_string_value(type);
                                        if(!strcasecmp(t, "a")) {
                                             file_av_1->opus = 1;
                                             file_av_1->vp9 = 0;
                                        } else if(!strcasecmp(t, "v")) {
                                             file_av_1->opus = 0;
                                             file_av_1->vp9 = 1;
                                        } else {
                                                printf("Unsupported recording type '%s' in info header...\n", t);
                                                exit(1);
                                        }
                                        /* What codec was used? */
                                        json_t *codec = json_object_get(info, "c");
                                        if(!codec || !json_is_string(codec)) {
                                                printf("Missing recording codec in info header...\n");
                                                exit(1);
                                        }
                                        const char *c = json_string_value(codec);
                                        if(!strcasecmp(c, "opus")) {
                                             file_av_1->opus = 1;
                                             file_av_1->vp9 = 0;
                                            if(extension && strcasecmp(extension, ".webm")) {
                                                printf("Opus RTP packets can only be converted to a .opus file\n");
                                                exit(1);
                                            }
                                        } else if(!strcasecmp(c, "vp9")) {
                                             file_av_1->opus = 0;
                                             file_av_1->vp9 = 1;
						if(extension && strcasecmp(extension, ".webm")) {
							printf("VP9 RTP packets can only be converted to a .webm file\n");
							exit(1);
						}
					} else {
                                            printf("The post-processor only supports Opus and G.711 audio for now (was '%s')...\n", c);
                                            exit(1);
                                        }
                                        /* When was the file created? */
                                        json_t *created = json_object_get(info, "s");
                                        if(!created || !json_is_integer(created)) {
                                                printf("Missing recording created time in info header...\n");
                                                exit(1);
                                        }
                                        file_av_1->c_time = json_integer_value(created);
                                        /* When was the first frame written? */
                                        json_t *written = json_object_get(info, "u");
                                        if(!written || !json_is_integer(written)) {
                                                printf("Missing recording written time in info header...\n");
                                                exit(1);
                                        }
                                        file_av_1->w_time = json_integer_value(written);
                                        /* Summary */
                                        printf("This is %s recording:\n", file_av_1->vp9 ? "a video" : "an audio");
                                        printf("  -- Codec:   %s\n", c);
                                        printf("  -- Created: %"SCNi64"\n", file_av_1->c_time);
                                        printf("  -- Written: %"SCNi64"\n", file_av_1->w_time);
                                }
                        } else {
                                printf("Invalid header...\n");
                                exit(1);
                        }        
                        /* Skip data for now */
                        file_av_1->offset += len;
                        //printf("%i %i",file_av_1->offset, file_av_1->fsize);
                }
                 //printf(" Hola \n");
                file_av_1 = file_av_1->next;
            }
            file_combine_1 = file_combine_1->next;
        }
	if(!working)
		exit(0);

	uint64_t max32 = UINT32_MAX;
	/* Start loop */
        file_combine_1 = file_combine_list_1->head;
        for(j = 0; j < i; j++) {
            file_av *file_av_1 = file_combine_1->file_av_list_1->head;
            for(m = 0; m < 2; m++) {
                working = 1;
                file_av_1->offset = 0;
                file_av_1->last_ts = 0;
                file_av_1->reset = 0;
                file_av_1->times_resetted = 0;
                file_av_1->post_reset_pkts = 0;
                int bytes = 0, skip = 0;
                uint16_t len = 0;
                char prebuffer[1500];
                memset(prebuffer, 0, 1500);
                while(working && file_av_1->offset < file_av_1->fsize) {
                        /* Read frame header */
                        skip = 0;
                        fseek(file_av_1->file, file_av_1->offset, SEEK_SET);
                        bytes = fread(prebuffer, sizeof(char), 8, file_av_1->file);
                        if(bytes != 8 || prebuffer[0] != 'M') {
                                /* Broken packet? Stop here */
                                break;
                        }
                        prebuffer[8] = '\0';
                        //printf("Header: %s\n", prebuffer);
                        file_av_1->offset += 8;
                        bytes = fread(&len, sizeof(uint16_t), 1, file_av_1->file);
                        len = ntohs(len);
                        //printf("  -- Length: %"SCNu16"\n", len);
                        file_av_1->offset += 2;
                        if(prebuffer[1] == 'J' || ( len < 12)) {
                                /* Not RTP, skip */
                                printf("  -- Not RTP, skipping\n");
                                file_av_1->offset += len;
                                continue;
                        }
                        if(len > 2000) {
                                /* Way too large, very likely not RTP, skip */
                                printf("  -- Too large packet (%d bytes), skipping\n", len);
                                file_av_1->offset += len;
                                continue;
                        }
                        /* Only read RTP header */
                        bytes = fread(prebuffer, sizeof(char), 16, file_av_1->file);
                        janus_pp_rtp_header *rtp = (janus_pp_rtp_header *)prebuffer;
                        //printf("  -- RTP packet (ssrc=%"SCNu32", pt=%"SCNu16", ext=%"SCNu16", seq=%"SCNu16", ts=%"SCNu32")\n",
                        //                ntohl(rtp->ssrc), rtp->type, rtp->extension, ntohs(rtp->seq_number), ntohl(rtp->timestamp));
                        if(rtp->csrccount) {
                                printf("  -- -- Skipping CSRC list\n");
                                skip += rtp->csrccount*4;
                        }
                        if(rtp->extension) {
                                janus_pp_rtp_header_extension *ext = (janus_pp_rtp_header_extension *)(prebuffer+12);
                                printf("  -- -- RTP extension (type=%"SCNu16", length=%"SCNu16")\n",
                                        ntohs(ext->type), ntohs(ext->length));
                                skip += 4 + ntohs(ext->length)*4;
                        }
                        /* Generate frame packet and insert in the ordered list */
                        janus_pp_frame_packet *p = (janus_pp_frame_packet *)g_malloc0(sizeof(janus_pp_frame_packet));
                        if(p == NULL) {
                                printf("Memory error!\n");
                                return -1;
                        }
                        p->seq = ntohs(rtp->seq_number);
                        p->pt = rtp->type;
                        /* Due to resets, we need to mess a bit with the original timestamps */
                        if(file_av_1->last_ts == 0) {
                                /* Simple enough... */
                                p->ts = ntohl(rtp->timestamp);
                        } else {
                                /* Is the new timestamp smaller than the next one, and if so, is it a timestamp reset or simply out of order? */
                                gboolean late_pkt = FALSE;
                                if(ntohl(rtp->timestamp) < file_av_1->last_ts && (file_av_1->last_ts-ntohl(rtp->timestamp) > 2*1000*1000*1000)) {
                                        if(file_av_1->post_reset_pkts > 1000) {
                                                file_av_1->reset = ntohl(rtp->timestamp);
                                                printf("Timestamp reset: %"SCNu32"\n", file_av_1->reset);
                                                file_av_1->times_resetted++;
                                                file_av_1->post_reset_pkts = 0;
                                        }
                                } else if(ntohl(rtp->timestamp) > file_av_1->reset && ntohl(rtp->timestamp) > file_av_1->last_ts &&
                                                (ntohl(rtp->timestamp)-file_av_1->last_ts > 2*1000*1000*1000)) {
                                        if(file_av_1->post_reset_pkts < 1000) {
                                                printf("Late pre-reset packet after a timestamp reset: %"SCNu32"\n", ntohl(rtp->timestamp));
                                                late_pkt = TRUE;
                                                file_av_1->times_resetted--;
                                        }
                                } else if(ntohl(rtp->timestamp) < file_av_1->reset) {
                                        if(file_av_1->post_reset_pkts < 1000) {
                                                printf("Updating latest timestamp reset: %"SCNu32" (was %"SCNu32")\n", ntohl(rtp->timestamp), file_av_1->reset);
                                                file_av_1->reset = ntohl(rtp->timestamp);
                                        } else {
                                                file_av_1->reset = ntohl(rtp->timestamp);
                                                printf("Timestamp reset: %"SCNu32"\n", file_av_1->reset);
                                                file_av_1->times_resetted++;
                                                file_av_1->post_reset_pkts = 0;
                                        }
                                }
                                /* Take into account the number of resets when setting the internal, 64-bit, timestamp */
                                p->ts = (file_av_1->times_resetted*max32)+ntohl(rtp->timestamp);
                                if(late_pkt)
                                        file_av_1->times_resetted++;
                        }
                        p->len = len;
                        p->drop = 0;
                        if(rtp->padding) {
                                /* There's padding data, let's check the last byte to see how much data we should skip */
                                fseek(file_av_1->file, file_av_1->offset + len - 1, SEEK_SET);
                                bytes = fread(prebuffer, sizeof(char), 1, file_av_1->file);
                                uint8_t padlen = (uint8_t)prebuffer[0];
                                printf("Padding at sequence number %hu: %d/%d\n",
                                        ntohs(rtp->seq_number), padlen, p->len);
                                p->len -= padlen;
                                if((p->len - skip - 12) <= 0) {
                                        /* Only padding, take note that we should drop the packet later */
                                        p->drop = 1;
                                        printf("  -- All padding, marking packet as dropped\n");
                                }
                        }
                        file_av_1->last_ts = ntohl(rtp->timestamp);
                        file_av_1->post_reset_pkts++;
                        /* Fill in the rest of the details */
                        p->offset = file_av_1->offset;
                        p->skip = skip;
                        p->next = NULL;
                        p->prev = NULL;
                        if(file_av_1->list == NULL) {
                                /* First element becomes the list itself (and the last item), at least for now */
                                file_av_1->list = p;
                                file_av_1->last = p;
                        } else {
                                /* Check where we should insert this, starting from the end */
                                int added = 0;
                                janus_pp_frame_packet *tmp = file_av_1->last;
                                while(tmp) {
                                        if(tmp->ts < p->ts) {
                                                /* The new timestamp is greater than the last one we have, append */
                                                added = 1;
                                                if(tmp->next != NULL) {
                                                        /* We're inserting */
                                                        tmp->next->prev = p;
                                                        p->next = tmp->next;
                                                } else {
                                                        /* Update the last packet */
                                                        file_av_1->last = p;
                                                }
                                                tmp->next = p;
                                                p->prev = tmp;
                                                break;
                                        } else if(tmp->ts == p->ts) {
                                                /* Same timestamp, check the sequence number */
                                                if(tmp->seq < p->seq && (abs(tmp->seq - p->seq) < 10000)) {
                                                        /* The new sequence number is greater than the last one we have, append */
                                                        added = 1;
                                                        if(tmp->next != NULL) {
                                                                /* We're inserting */
                                                                tmp->next->prev = p;
                                                                p->next = tmp->next;
                                                        } else {
                                                                /* Update the last packet */
                                                                file_av_1->last = p;
                                                        }
                                                        tmp->next = p;
                                                        p->prev = tmp;
                                                        break;
                                                } else if(tmp->seq > p->seq && (abs(tmp->seq - p->seq) > 10000)) {
                                                        /* The new sequence number (resetted) is greater than the last one we have, append */
                                                        added = 1;
                                                        if(tmp->next != NULL) {
                                                                /* We're inserting */
                                                                tmp->next->prev = p;
                                                                p->next = tmp->next;
                                                        } else {
                                                                /* Update the last packet */
                                                                file_av_1->last = p;
                                                        }
                                                        tmp->next = p;
                                                        p->prev = tmp;
                                                        break;
                                                }
                                        }
                                        /* If either the timestamp ot the sequence number we just got is smaller, keep going back */
                                        tmp = tmp->prev;
                                }
                                if(!added) {
                                        /* We reached the start */
                                        p->next = file_av_1->list;
                                        file_av_1->list->prev = p;
                                        file_av_1->list = p;
                                }
                        }
                        /* Skip data for now */
                        file_av_1->offset += len;
                        file_av_1->count++;
                }
                printf("Counted file %"SCNu32" RTP packets\n", file_av_1->count);
                file_av_1 = file_av_1->next;
            }
            file_combine_1 = file_combine_1->next;
        }
        if(!working)
            exit(0);
        file_combine_1 = file_combine_list_1->head;
	for (j = 0; j<i; j++) {
            file_av *file_av_1 = file_combine_1->file_av_list_1->head;
            for(m = 0;  m<2; m++) {
                janus_pp_frame_packet *tmp = file_av_1->list;
                file_av_1->count = 0;
                while(tmp) {
                        file_av_1->count++;
                    //    printf("[%10lu][%4d] seq=%"SCNu16", ts=%"SCNu64", time=%"SCNu64"s\n", tmp->offset, tmp->len, tmp->seq, tmp->ts, (tmp->ts-file_av_1->list->ts)/90000);
                        tmp = tmp->next;
                }
                printf("Counted %"SCNu32" frame packets in file\n", file_av_1->count);
                if(file_av_1->vp9)
                    janus_pp_webm_preprocess(file_av_1);
                
                file_av_1 = file_av_1->next;
            }
        }
        
	if(janus_pp_webm_create(destination) < 0) {
            printf("Error creating .webm file...\n");
            exit(1);
	}
        
        if(janus_pp_webm_process(file_combine_list_1, &working) < 0) {
            printf("Error processing %s RTP frames...\n");
	}
        /*
	janus_pp_webm_close();
	fclose(file);
	
	file = fopen(destination, "rb");
	if(file == NULL) {
		printf("No destination file %s??\n", destination);
	} else {
		fseek(file, 0L, SEEK_END);
		fsize = ftell(file);
		fseek(file, 0L, SEEK_SET);
		printf("%s is %zu bytes\n", destination, fsize);
		fclose(file);
	}
	janus_pp_frame_packet *temp = list, *next = NULL;
	while(temp) {
		next = temp->next;
		g_free(temp);
		temp = next;
	}
        */
	printf("Bye!\n");
	return 0;
}




