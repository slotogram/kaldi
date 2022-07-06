// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave-reader.h"
#include "online/online-audio-source.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include <locale.h>
#include <windows.h>
#include "lat/confidence.h"
#include "lat/word-align-lattice.h"
#include <stdio.h>

using namespace std;
namespace kaldi {
	static bool do_endpointing = false;
	static bool online = true;
	static OnlineEndpointConfig endpoint_opts;
	
	//static SingleUtteranceNnet3Decoder* decoder_;
	//static OnlineNnet2FeaturePipeline* feature_pipeline_;
	//static OnlineSilenceWeighting* silence_weighting_;
	//static OnlineIvectorExtractorAdaptationState* adaptation_state_;
	static TransitionModel trans_model;
	static nnet3::AmNnetSimple am_nnet;
	static nnet3::DecodableNnetSimpleLoopedInfo* decodable_info;
	static nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
	static LatticeFasterDecoderConfig decoder_opts;
	static OnlineNnet2FeaturePipelineInfo* feature_info;
	static fst::Fst<fst::StdArc> *decode_fst;
	static fst::LookaheadFst<fst::StdArc, int32> *decode_fst_ = nullptr;
	static fst::StdOLabelLookAheadFst *hcl_fst_ = nullptr; //StdOLabelLookAheadFst
	static fst::NGramFst<fst::StdArc> *g_fst_ = nullptr;
	static vector<int32> disambig_;
	static const fst::SymbolTable *word_syms = NULL;
	static kaldi::WordBoundaryInfo *winfo_ = nullptr;
	static const int32 kSampleFreq = 8000; // Input sampling frequency is fixed to 8KHz
	static int32 chunk_length;

	FILE *stream;

	std::wstring string_to_wstring(string str) { //TODO: check if buf and sss not needed
		std::wstring convertedString;
		int requiredSize = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, 0, 0); 
		
		/*char* buf = new char[requiredSize*2 - 1];
		str.copy(buf, str.size());

		wchar_t* sss = new wchar_t[requiredSize];
		for (int i = 0; i<requiredSize; i++) {
			sss[i] = (((wchar_t)(buf[i * 2])) & 255) << 8 | ((buf[i * 2 + 1]) & 255);
		}
		
		std::wstring wsttt = std::wstring(sss);
		*/
		if (requiredSize > 0) {
			std::vector<wchar_t> buffer(requiredSize);
			MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &buffer[0], requiredSize);
			convertedString.assign(buffer.begin(), buffer.end() - 1);
		}
		//delete buf, sss;
		return convertedString;
	}

	std::string encode_1251(std::wstring &wstr)
	{
		//if (wstr.empty()) return std::string();
		int size_needed = WideCharToMultiByte(1251, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
		std::string strTo(size_needed, 0);
		WideCharToMultiByte(1251, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
		return strTo;
	}


void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";
  //std::ofstream out;          // поток для записи
  //out.open("hello.txt");
  if (word_syms != NULL) {
    //std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      
	//  out << s << ' '; 
	  s = encode_1251(string_to_wstring(s)); // kaldi формирует строку в utf-8, но система не распознает ее как utf-8. 
 										 // как эти изменения отразятся, если тут будет не русский текст???
      std::cerr << s << ' ';
	  std::cout << s << ' ';
    }
    //std::cerr << std::endl;
  }
//  out.close();
}


void init_kaldi_nnet3()
{
	using namespace kaldi;
	using namespace fst;
	using namespace std::chrono;
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;

	fprintf(stdout, "Initializing neural net\n"); fflush(stdout);

	const char *usage =
		"Reads in wav file(s) and simulates online decoding with neural nets\n"
		"(nnet3 setup), with optional iVector-based speaker adaptation and\n"
		"optional endpointing.  Note: some configuration values and inputs are\n"
		"set via config files whose filenames are passed as options\n"
		"\n"
		"Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
		"<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
		"The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
		"you want to decode utterance by utterance.\n";

	ParseOptions po(usage);

	std::string word_syms_rxfilename;

	// feature_opts includes configuration for the iVector adaptation,
	// as well as the basic features.
	
	OnlineNnet2FeaturePipelineConfig feature_opts;

	endpoint_opts.silence_phones = "1:2:3:4:5:6:7:8:9:10";
	endpoint_opts.rule2.min_trailing_silence = 0.25;
	endpoint_opts.rule3.min_trailing_silence = 0.5;
	//endpoint_opts.
	BaseFloat chunk_length_secs = 0.18;


	po.Register("chunk-length", &chunk_length_secs,
		"Length of chunk size in seconds, that we process.  Set to <= 0 "
		"to use all input in one chunk.");
	po.Register("word-symbol-table", &word_syms_rxfilename,
		"Symbol table for words [for debug output]");
	po.Register("do-endpointing", &do_endpointing,
		"If true, apply endpoint detection");
	po.Register("online", &online,
		"You can set this to false to disable online iVector estimation "
		"and have all the data for each utterance used, even at "
		"utterance start.  This is useful where you just want the best "
		"results and don't care about online operation.  Setting this to "
		"false has the same effect as setting "
		"--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
		"in the file given to --ivector-extraction-config, and "
		"--chunk-length=-1.");
	po.Register("num-threads-startup", &g_num_threads,
		"Number of threads used when initializing iVector extractor.");

	feature_opts.Register(&po);
	decodable_opts.Register(&po);
	decoder_opts.Register(&po);
	endpoint_opts.Register(&po);

	//
	int argc = 14;
	char *argv[] = { (char*)"foo.exe",
		//(char*)"--do-endpointing=true",
		(char*)"--word-symbol-table=exp/tdnn/graph/words.txt",
		(char*)"--frame-subsampling-factor=3",
		(char*)"--frames-per-chunk=51",
		(char*)"--acoustic-scale=1.0",
		(char*)"--beam=12.0",
		(char*)"--lattice-beam=6.0",
		(char*)"--max-active=10000",
		(char*)"--config=exp/tdnn/conf/online.conf",
		(char*)"exp/tdnn/final.mdl",
		(char*)"exp/tdnn/graph/HCLG.fst",
		(char*)"ark:decoder-test.utt2spk",
		(char*)"scp:decoder-test.scp",
		(char*)"ark:-",
		NULL };

	po.Read(argc, argv);

	if (po.NumArgs() != 5) {
		po.PrintUsage();
		return;
	}

	std::string nnet3_rxfilename = po.GetArg(1),
		fst_rxfilename = po.GetArg(2),
		spk2utt_rspecifier = po.GetArg(3),
		wav_rspecifier = po.GetArg(4),
		clat_wspecifier = po.GetArg(5);

	feature_info = new OnlineNnet2FeaturePipelineInfo(feature_opts);

	if (!online) {
		feature_info->ivector_extractor_info.use_most_recent_ivector = true;
		feature_info->ivector_extractor_info.greedy_ivector_extractor = true;
		chunk_length_secs = -1.0;
	}

	//am_nnet = new nnet3::AmNnetSimple();
	{
		bool binary;
		Input ki(nnet3_rxfilename, &binary);
		trans_model.Read(ki.Stream(), binary);
		am_nnet.Read(ki.Stream(), binary);
		nnet3::SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
		nnet3::SetDropoutTestMode(true, &(am_nnet.GetNnet()));
		nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
	}

	// this object contains precomputed stuff that is used by all decodable
	// objects.  It takes a pointer to am_nnet because if it has iVectors it has
	// to modify the nnet to accept iVectors at intervals.
	decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts,
		&am_nnet);

	const char* hcl_fst_rxfilename_ = "exp/tdnn/graph/HCLr.fst";
	const char* g_fst_rxfilename_ = "exp/tdnn/graph/Gr.fst";
	const char* disambig_rxfilename_ = "exp/tdnn/graph/disambig_tid.int";

	struct stat buffer;

	if (stat(fst_rxfilename.c_str(), &buffer) == 0) {
		decode_fst = ReadFstKaldiGeneric(fst_rxfilename);
	}
	else
	{
		//hcl_fst_ = fst::StdFst::Read(hcl_fst_rxfilename_);
		hcl_fst_ = fst::StdOLabelLookAheadFst::Read(hcl_fst_rxfilename_);

		g_fst_ = fst::NGramFst<fst::StdArc>::Read(g_fst_rxfilename_);
		ReadIntegerVectorSimple(disambig_rxfilename_, &disambig_);
		decode_fst_ = LookaheadComposeFst(*hcl_fst_, *g_fst_, disambig_);

	}


	if (g_fst_ && g_fst_->OutputSymbols()) {
		word_syms = g_fst_->OutputSymbols();
	}
	else
		//fst::SymbolTable *word_syms = NULL;
		if (word_syms_rxfilename != "")
			if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
				KALDI_ERR << "Could not read symbol table from file "
				<< word_syms_rxfilename;

	const char* winfo_rxfilename_ = "exp/tdnn/graph/phones/word_boundary.int";

	if (stat(winfo_rxfilename_, &buffer) == 0) {
		KALDI_LOG << "Loading winfo " << winfo_rxfilename_;
		kaldi::WordBoundaryInfoNewOpts opts;
		winfo_ = new kaldi::WordBoundaryInfo(opts, winfo_rxfilename_);
	}


	if (chunk_length_secs > 0) {
		chunk_length = int32(feature_info->GetSamplingFrequency() * chunk_length_secs);
		if (chunk_length == 0) chunk_length = 1;
	}
	else {
		chunk_length = std::numeric_limits<int32>::max();
	}
}
OnlinePaSource* init_kaldi_portaudio()
{
	do_endpointing = true;
	online = true;
	init_kaldi_nnet3();


	// Timeout interval for the PortAudio source
	const int32 kTimeout = 500; // half second

	// PortAudio's internal ring buffer size in bytes
	const int32 kPaRingSize = 32768;
	// Report interval for PortAudio buffer overflows in number of feat. batches
	const int32 kPaReportInt = 4;
	//OnlinePaSource au_src(kTimeout, kSampleFreq, kPaRingSize, kPaReportInt);
	
	OnlinePaSource* au_src = new OnlinePaSource(kTimeout, feature_info->GetSamplingFrequency(), kPaRingSize, kPaReportInt);
	return au_src;
}
}



int main(int argc, char *argv[]) {
	using namespace kaldi;
	using namespace fst;
	
	// Reassign "stderr" to "freopen.out":
	stream = freopen("Errors.log", "w", stderr); // C4996
	// Note: freopen is deprecated; consider using freopen_s instead

	if (stream == NULL)
		fprintf(stdout, "error on freopen\n");
	else
	{
		fprintf(stdout, "successfully reassigned\n"); fflush(stdout);
		fprintf(stream, "SRC started'\n");
	}
	
	
	
	try {
    

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
	setlocale (LC_ALL,"Russian");

	

	//SetConsoleOutputCP(CP_UTF8);
	//SetConsoleCP(CP_UTF8);

	OnlinePaSource *au_src = init_kaldi_portaudio();

	int32 num_done = 0, num_err = 0;
	double tot_like = 0.0;
	int64 num_frames = 0;

	OnlineTimingStats timing_stats;


	std::cerr << std::endl;
    for (; ; ) {
		//OnlineNnet2FeaturePipelineInfo feature_info1 = *feature_info;
		//OnlineIvectorExtractionInfo* tst = &feature_info->ivector_extractor_info;
		OnlineIvectorExtractorAdaptationState adaptation_state(
			(feature_info->ivector_extractor_info)); //feature_info->ivector_extractor_info

		string utt = "test";
		while (true) {
			OnlineNnet2FeaturePipeline feature_pipeline(*feature_info);
			feature_pipeline.SetAdaptationState(adaptation_state);

			OnlineSilenceWeighting silence_weighting(
				trans_model,
				feature_info->silence_weighting_config,
				decodable_opts.frame_subsampling_factor);

			SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
				*decodable_info,
				hcl_fst_ ? *decode_fst_ : *decode_fst, &feature_pipeline);
        //OnlineTimer decoding_timer(utt);

                //int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;
		std::cerr << "Audio IN started" << std::endl;
		fprintf(stdout, "\nYou can speak now\n"); fflush(stdout);
        while (true) {
 
		  // Prepare the input audio samples
		  Vector<BaseFloat> wave_part(chunk_length);
		  bool ans = au_src->Read(&wave_part);
		  
		  //std::cerr << wave_part.Min() << ' ' << wave_part.Max();
          feature_pipeline.AcceptWaveform(feature_info->GetSamplingFrequency(), wave_part);
		  
          //samp_offset += chunk_length;
          //decoding_timer.WaitUntil(samp_offset / kSampleFreq);
          //if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
          //  feature_pipeline.InputFinished();
          //}
		  
          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
          }
		  
          decoder.AdvanceDecoding();
		  
		  
          if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
			  feature_pipeline.InputFinished(); 
			  break;
          }
        }
		std::cerr << '\'';
        decoder.FinalizeDecoding();
		
        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                     &num_frames, &tot_like);
		std::cerr << '\'' << std::endl;
		fflush(stream);
        //decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);
		
        // we want to output the lattice with un-scaled acoustics.
        //BaseFloat inv_acoustic_scale =
        //    1.0 / decodable_opts.acoustic_scale;
        //ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

        //clat_writer.Write(utt, clat);
        //KALDI_LOG << "Decoded utterance " << utt;
        num_done++;
      }
    }
    //timing_stats.Print(online); 

	/* KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
	*/
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
	fclose(stream);
    return -1;
  }
} // main()
