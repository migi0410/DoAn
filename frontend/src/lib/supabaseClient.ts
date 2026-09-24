import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://qwprbxxxbvueozffqhdd.supabase.co';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF3cHJieHh4YnZ1ZW96ZmZxaGRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4OTU5NzA4MywiZXhwIjoyMTA1MTczMDgzfQ.PwuArhKLDbeQrg_SLVSFxXqhoIUGJRDbawyVnayrooA';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
