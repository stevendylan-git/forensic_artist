export interface FeatureAttribute {
  value: string;
  confidence: number;
  stated: boolean;
  locked: boolean;
  notes?: string;
}

export interface FaceSpec {
  gender_presentation: FeatureAttribute;
  approximate_age: FeatureAttribute;
  face_shape: FeatureAttribute;
  skin_tone_complexion: FeatureAttribute;
  hair_style_color: FeatureAttribute;
  eyebrows: FeatureAttribute;
  eyes_shape_color: FeatureAttribute;
  nose_shape: FeatureAttribute;
  mouth_lips: FeatureAttribute;
  jawline_chin: FeatureAttribute;
  facial_hair: FeatureAttribute;
  distinguishing_marks: FeatureAttribute;
  accessories: FeatureAttribute;
}

export interface HealthResponse {
  status: string;
  environment: string;
  database_connected: boolean;
  gemini_configured: boolean;
  gemini_text_model: string;
  gemini_image_model: string;
  version: string;
  error?: string;
}

export interface AuditEntry {
  id: number;
  timestamp: string;
  case_id?: string;
  user_id: string;
  client_ip?: string;
  endpoint: string;
  http_method: string;
  status_code: number;
  action_type: string;
  detail?: string;
  payload_hash: string;
  previous_hash: string;
  entry_hash: string;
}

export interface AuditVerificationResult {
  is_valid: boolean;
  total_entries: number;
  broken_at_entry_id?: number;
  expected_hash?: string;
  actual_hash?: string;
  message: string;
}
