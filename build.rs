use std::{env, fs, path::PathBuf, process::Command};

const VALUE_NET: &str = "tyltaks_2560_2";
const POLICY_NET: &str = "policy002";

const DOWNLOAD_URL: &str = "https://github.com/Ciekce/takson-nets/releases/download";

fn main() {
    println!("cargo::rerun-if-env-changed=EVALFILE");
    println!("cargo::rerun-if-env-changed=POLICYFILE");

    if let Ok(p) = env::var("EVALFILE") {
        eprintln!("using value: {p}");
        println!("cargo::rustc-env=EVALFILE={}", fs::canonicalize(p).unwrap().display());
        return;
    }

    if let Ok(p) = env::var("POLICYFILE") {
        eprintln!("using policy: {p}");
        println!("cargo::rustc-env=POLICYFILE={}", fs::canonicalize(p).unwrap().display());
        return;
    }

    //TODO deduplicate

    let value_out = PathBuf::from(env::var("OUT_DIR").unwrap()).join(format!("{VALUE_NET}.val"));

    if value_out.exists() {
        eprintln!("reusing {VALUE_NET}");
    } else {
        let url = format!("{DOWNLOAD_URL}/{VALUE_NET}/{VALUE_NET}.val");
        eprintln!("downloading {VALUE_NET} from {url}");
        let status = Command::new("curl")
            .args(["-sSfL", "-o"])
            .arg(&value_out)
            .arg(&url)
            .status()
            .unwrap();
        assert!(status.success(), "failed to download {VALUE_NET} from {url}");
    }

    println!("cargo::rustc-env=EVALFILE={}", value_out.display());

    let policy_out = PathBuf::from(env::var("OUT_DIR").unwrap()).join(format!("{POLICY_NET}.pol"));

    if policy_out.exists() {
        eprintln!("reusing {POLICY_NET}");
    } else {
        let url = format!("{DOWNLOAD_URL}/{POLICY_NET}/{POLICY_NET}.pol");
        eprintln!("downloading {POLICY_NET} from {url}");
        let status = Command::new("curl")
            .args(["-sSfL", "-o"])
            .arg(&policy_out)
            .arg(&url)
            .status()
            .unwrap();
        assert!(status.success(), "failed to download {POLICY_NET} from {url}");
    }

    println!("cargo::rustc-env=POLICYFILE={}", policy_out.display());
}
