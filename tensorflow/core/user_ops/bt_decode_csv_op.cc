/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/parsing_ops.cc.
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

class BtDecodeCSVOp : public OpKernel {
 public:
  explicit BtDecodeCSVOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

//    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
//    OP_REQUIRES_OK(ctx, ctx->GetAttr("field_delim", &delim));

  }


  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));

    OpOutputList itm_fea_lst;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("itm_fea_lst", &itm_fea_lst));

    auto records_t = records->flat<string>();
    int64 records_size = records_t.size();
    int64 itmNum = num_outputs();
    std::vector<int64> sizArr;
    for (int64 i = 0; i < records_size; ++i) {
      const StringPiece record(records_t(i));
      std::vector< std::vector<float> > feas;
      splitByTab(ctx, record, feas);
      if(i==0) {//allocate memory space
    	OP_REQUIRES(ctx, itmNum == feas.size(),
							  errors::InvalidArgument(
								  "The number of items contained in each line must be equal to the num_itm attribute!"));
    	Tensor* lbl = nullptr;
        itm_fea_lst.allocate(0,TensorShape({records_size}), &lbl);
        sizArr.push_back(1);
	    for (size_t fn = 1; fn < itmNum; ++fn) {//BTBT TODO this doesn't work in sparse case, because the num of features is not the same even in the same dim
	      int64 len = feas[fn].size();
	      Tensor* out = nullptr;
	      itm_fea_lst.allocate(fn, TensorShape({records_size,len}), &out);
	      sizArr.push_back(len);
	    }

      }
      itm_fea_lst[0]->flat<float>()(i) = feas[0][0];//label
	  for(int64 f=1; f<itmNum; ++f) {
    	  std::vector<float> onefeas = feas[f];
    	  int64 rangeBegin = i * sizArr[f];//BTBT TODO this doesn't work in sparse case, because the num of features is not the same even in the same dim
    	  auto flat = itm_fea_lst[f]->flat<float>();
    	  for(int64 inf=0; inf<sizArr[f]; ++inf) {
    		  flat(rangeBegin+inf) =  onefeas[inf];
    	  }
      }
    }
  }

 private:
  char space_=' ';
  char collon_=':';
  char tab_ = '\t';

	void splitByTab(OpKernelContext* ctx, StringPiece input, std::vector< std::vector<float> > &result) {
		int64 current_idx = 0;
		int convertCnt = 0;
		if (!input.empty()) {
		  while (static_cast<size_t>(current_idx) < input.size()) {
			if (input[current_idx] == '\n' || input[current_idx] == '\r') {
			  current_idx++;
			  continue;
			}

			// This is the body of the field;
			string field;
			  while (static_cast<size_t>(current_idx) < input.size() &&
					 input[current_idx] != tab_) {
				OP_REQUIRES(ctx, input[current_idx] != '"' &&
								 input[current_idx] != '\n' &&
								 input[current_idx] != '\r',
						errors::InvalidArgument(
							"Unquoted fields cannot have quotes/CRLFs inside"));
				field += input[current_idx];
				current_idx++;
			  }

			  // Go to next field or the end
			  current_idx++;

			if(!field.empty()) {
				convertCnt++;
				if(1==convertCnt) {//BTBT TODO the label field must change to be a int scala tensor
					float i = -1;
					OP_REQUIRES(ctx, strings::safe_strtof(field.c_str(), &i),
                          errors::InvalidArgument("Label field ", field, " is not a valid float! "));
					std::vector<float> lbl;
					lbl.push_back(i);
					result.push_back(lbl);
				} else {
					std::vector<float> vals;
					splitByComma(ctx, field,vals);
					result.push_back(vals);
				}
			 }
		  }
		}
	//	  if (input[input.size() - 1] == comma_) result.push_back(string());
	}

	void splitByComma(OpKernelContext* ctx, string &input, std::vector<float> &result) {
		int64 current_idx = 0;
		if (!input.empty()) {
		  while (static_cast<size_t>(current_idx) < input.size()) {
			if (input[current_idx] == '\n' || input[current_idx] == tab_ || input[current_idx] == '\r') {
			  current_idx++;
			  continue;
			}

			// This is the body of the field;
			string field;
			  while (static_cast<size_t>(current_idx) < input.size() &&
					 input[current_idx] != space_) {
				OP_REQUIRES(ctx, input[current_idx] != '"' &&
								 input[current_idx] != '\n' &&
								 input[current_idx] != tab_ &&
								 input[current_idx] != '\r',
						errors::InvalidArgument(
							"Unquoted fields cannot have quotes/CRLFs/tabs inside"));
				field += input[current_idx];
				current_idx++;
			  }

			  // Go to next field or the end
			  current_idx++;

			if(!field.empty()) {
				int64 k = -1;
				float v = -1;
				splitByCollon(ctx,field,k,v);
				OP_REQUIRES(ctx, -1 != k && -1 != v,
						errors::InvalidArgument(
							"Err happend when converting fea idx and val"));
				result.push_back(v);//BTBT TODO need to handle the feature index and treat example as sparse vector
			 }

		  }
		  // Check if the last field is missing
	//	  if (input[input.size() - 1] == comma_) result.push_back(string());
		}
	}


  void splitByCollon(OpKernelContext* ctx, string &input, int64 &k, float &v) {
		int current_idx = 0;
		int covertCount = 0;
		if (!input.empty()) {
		  while (static_cast<size_t>(current_idx) < input.size()) {
			if (input[current_idx] == '\n' || input[current_idx] == tab_ ||
					input[current_idx] == space_ || input[current_idx] == '\r') {
			  current_idx++;
			  continue;
			}
			// This is the body of the field;
			string field;
			  while (static_cast<size_t>(current_idx) < input.size() &&
				 input[current_idx] != collon_) {
				OP_REQUIRES(ctx, input[current_idx] != '"' &&
									 input[current_idx] != '\n' &&
									 input[current_idx] != space_ &&
									 input[current_idx] != tab_ &&
									 input[current_idx] != '\r',
							errors::InvalidArgument(
								"Unquoted fields cannot have quotes/CRLFs/tabs/space inside"));
				field += input[current_idx];
				current_idx++;
			  }

			  // Go to next field or the end
			  current_idx++;

			if(!field.empty()) {
				covertCount++;
				if(1 == covertCount) {
					int64 ik = -1;
					OP_REQUIRES(ctx, strings::safe_strto64(field.c_str(), &ik),
                          errors::InvalidArgument("Field ", field, " is not a valid int64! "));
					k = ik;
				} else if (2 == covertCount) {
					float iv = -1;
					OP_REQUIRES(ctx, strings::safe_strtof(field.c_str(), &iv),
                          errors::InvalidArgument("Field ", field, " is not a valid float! "));
					v = iv;
				} else {
					OP_REQUIRES(ctx, covertCount < 3,
							errors::InvalidArgument(
								"There can be only 1 field on the left/right of collon"));
				}
			 }
		  }
		}
	}


};

REGISTER_KERNEL_BUILDER(Name("BtDecodeCSV").Device(DEVICE_CPU), BtDecodeCSVOp);

REGISTER_OP("BtDecodeCSV")
    .Input("records: string")
    .Output("itm_fea_lst: num_itm * float")
	.Attr("num_itm: int >= 1")
    .Doc(R"doc(
_________________
For decoding the svm like formate which used by DSSM.
)doc");

}  // namespace tensorflow
