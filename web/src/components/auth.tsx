"use client";

/**
 * Authentication components - AuthProvider, LoginForm, UserMenu.
 * Consolidated into a single file for simplicity (KISS principle).
 */

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { Session, User } from "@supabase/supabase-js";
import { Auth } from "@supabase/auth-ui-react";
import { ThemeSupa } from "@supabase/auth-ui-shared";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { LogOut } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { Button } from "./ui";

// =============================================================================
// Auth Context
// =============================================================================

interface AuthContextType {
  session: Session | null;
  user: User | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  return (
    <AuthContext.Provider value={{ session, user, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

// =============================================================================
// Login Form
// =============================================================================

export function LoginForm() {
  const [origin, setOrigin] = useState<string>("");

  useEffect(() => {
    setOrigin(window.location.origin);
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-ncl-blue to-ncl-blue-dark p-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow-xl p-8">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-ncl-blue">Maritime Technical Support System (MTSS)</h1>
          <p className="text-ncl-gray mt-2">Sign in to access vessel issue history and technical knowledge base</p>
        </div>

        <Auth
          supabaseClient={supabase}
          appearance={{
            theme: ThemeSupa,
            variables: {
              default: {
                colors: {
                  brand: "#003A8F",
                  brandAccent: "#001F5B",
                  inputBackground: "white",
                  inputBorder: "#D1D3D4",
                  inputBorderHover: "#4F83CC",
                  inputBorderFocus: "#003A8F",
                },
                borderWidths: {
                  buttonBorderWidth: "1px",
                  inputBorderWidth: "1px",
                },
                radii: {
                  borderRadiusButton: "0.375rem",
                  buttonBorderRadius: "0.375rem",
                  inputBorderRadius: "0.375rem",
                },
              },
            },
          }}
          providers={["google", "github", "azure"]}
          redirectTo={origin}
          magicLink={true}
          view="sign_in"
        />
      </div>
    </div>
  );
}

// =============================================================================
// User Menu
// =============================================================================

export function UserMenu() {
  const { user, signOut } = useAuth();

  if (!user) return null;

  const initials = user.email
    ? user.email.slice(0, 2).toUpperCase()
    : "U";

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative h-10 w-10 rounded-full bg-ncl-blue-light"
        >
          <span className="text-sm font-medium text-white">{initials}</span>
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[200px] bg-white rounded-md shadow-lg border border-ncl-gray-light p-1 z-50"
          align="end"
          sideOffset={5}
        >
          <div className="px-3 py-2 border-b border-ncl-gray-light">
            <p className="text-sm font-medium text-ncl-blue-dark truncate">
              {user.email}
            </p>
          </div>
          <DropdownMenu.Item
            className="flex items-center gap-2 px-3 py-2 text-sm text-ncl-gray cursor-pointer hover:bg-ncl-gray-light/20 rounded-md outline-none"
            onSelect={() => signOut()}
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
