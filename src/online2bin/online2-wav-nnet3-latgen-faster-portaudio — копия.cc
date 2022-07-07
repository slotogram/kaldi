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
namespace kaldi {

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
    }
    //std::cerr << std::endl;
  }
//  out.close();
}

}



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
	setlocale (LC_ALL,"Russian");
	//SetConsoleOutputCP(CP_UTF8);
	//SetConsoleCP(CP_UTF8);


	// Time out interval for the PortAudio source
	const int32 kTimeout = 500; // half second
								// Input sampling frequency is fixed to 8KHz
	const int32 kSampleFreq = 8000;
	// PortAudio's internal ring buffer size in bytes
	const int32 kPaRingSize = 32768;
	// Report interval for PortAudio buffer overflows in number of feat. batches
	const int32 kPaReportInt = 4;

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
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    LatticeFasterDecoderConfig decoder_opts;
    OnlineEndpointConfig endpoint_opts;
	endpoint_opts.silence_phones = "1:2:3:4:5:6:7:8:9:10";
	//endpoint_opts.
    BaseFloat chunk_length_secs = 0.18;
    bool do_endpointing = false;
    bool online = true;

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

	//char *argv[] = { "foo.exe", "-x", "myfile", "-f", "myflag", NULL 	};
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        clat_wspecifier = po.GetArg(5);

    OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);


    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    //SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    //RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    //CompactLatticeWriter clat_writer(clat_wspecifier);

    OnlineTimingStats timing_stats;

	OnlinePaSource au_src(kTimeout, kSampleFreq, kPaRingSize, kPaReportInt);
	std::cerr << std::endl;

    for (; ; ) {
      OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);
	  string utt = "test";
      while (true) {
    
        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config,
            decodable_opts.frame_subsampling_factor);

        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);
        //OnlineTimer decoding_timer(utt);

        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(kSampleFreq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        //int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;
		std::cerr << "Audio IN started" << std::endl;

        while (true) {
 
		  // Prepare the input audio samples
		  Vector<BaseFloat> wave_part(chunk_length);
		  bool ans = au_src.Read(&wave_part);
		  
		  //std::cerr << wave_part.Min() << ' ' << wave_part.Max();
          feature_pipeline.AcceptWaveform(kSampleFreq, wave_part);
		  
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
    return -1;
  }
} // main()
