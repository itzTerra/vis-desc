// Bundled proto text used by the frontend to build protobufjs types.
// Keep this in sync with services/api/core/protos/spacy.proto
export const SPACY_PROTO = `syntax = "proto3";
package spacy;

message Token {
  int32 paragraph_id = 1;
  int32 sentence_id = 2;
  int32 within_sentence_id = 3;
  int32 token_id = 4;
  string text = 5;
  string pos = 6;
  string fine_pos = 7;
  string lemma = 8;
  string deprel = 9;
  int32 dephead = 10;
  string ner = 11;
  int32 start_byte = 12;
  int32 end_byte = 13;
  map<string, string> morph = 14;
  bool like_num = 15;
  bool is_stop = 16;
  string itext = 17;
  bool in_quote = 18;
  bool event = 19;
}

message SentToken {
  string text = 1;
  string pos_ = 2;
  string dep_ = 3;
  repeated SentToken children = 4;
}

message Sentence {
  SentToken root = 1;
  int32 start = 2;
  int32 end = 3;
}

message NounChunk {
  int32 start = 1;
  int32 end = 2;
  string text = 3;
}

message SpaCyContext {
  repeated Token tokens = 1;
  repeated Sentence sentences = 2;
  repeated NounChunk noun_chunks = 3;
}

message SpaCyContexts {
  repeated SpaCyContext contexts = 1;
}
`;

export default SPACY_PROTO;
