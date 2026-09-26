import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://rdtyboaufaqbvmnombwp.supabase.co';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJkdHlib2F1ZmFxYnZtbm9tYndwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTA0Mzk2MzksImV4cCI6MjEwNjAxNTYzOX0.O4KwKbDzbgRIczD0kIJyiQXzypCWwMMEGEPVCr_Y-vs';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
