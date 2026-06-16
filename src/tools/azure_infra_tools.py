"""Azure infrastructure prerequisite scanner for MLOps pipeline setup.

Checks the six resources that must exist before AzDO pipelines can run end-to-end:
  1. Azure Resource Group
  2. Azure Container Registry (ACR) — CI pipeline pushes Docker images here
  3. AML Workspace — training / batch inference / monitoring runs here
  4. AML Compute Cluster — executes the AML jobs
  5. AzDO → Azure RM service connection — lets CD pipelines deploy to Azure
  6. AzDO → ACR service connection — lets CI pipelines push Docker images

Each check accepts config values from the project profile first, falling back to
global .env settings — so no manual .env edits are needed after auto-discovery.

All Azure SDK calls are lazy-imported: if a SDK package is not installed the
check reports "configured (SDK check skipped)" rather than crashing.
"""
from typing import List, Literal

from pydantic import BaseModel

from src.config.settings import get_settings
from src.shared.logging import get_logger

logger = get_logger(__name__)


class PrereqItem(BaseModel):
    name: str
    status: Literal["ok", "missing_config", "not_found", "error"]
    detail: str = ""
    required_for: str = ""
    fix: str = ""      # actionable instructions shown to the user when status != ok


class InfraReport(BaseModel):
    checks: List[PrereqItem]

    @property
    def all_ok(self) -> bool:
        return all(c.status == "ok" for c in self.checks)

    def markdown(self) -> str:
        lines = [
            "## Infrastructure Prerequisites\n",
            "| | Resource | Status | Required for |",
            "|---|---|---|---|",
        ]
        for item in self.checks:
            icon = "✅" if item.status == "ok" else ("⚠️" if item.status == "missing_config" else "❌")
            label = {
                "ok": "Ready",
                "missing_config": f"Not configured — {item.detail}" if item.detail else "Not configured in .env",
                "not_found": f"Not found — {item.detail[:80]}" if item.detail else "Not found in Azure",
                "error": f"Check failed — {item.detail[:80]}" if item.detail else "Check failed",
            }.get(item.status, item.status)
            lines.append(f"| {icon} | **{item.name}** | {label} | {item.required_for} |")

        ok = sum(1 for c in self.checks if c.status == "ok")
        lines.append(f"\n**{ok}/{len(self.checks)} prerequisites met.**")
        if not self.all_ok:
            missing = [c.name for c in self.checks if c.status != "ok"]
            lines.append(f"\n> ⚠️ Set up the missing resources before triggering pipelines: {', '.join(missing)}")
        return "\n".join(lines)


class DiscoveredConfig(BaseModel):
    """Result of auto-discovering Azure settings from AzDO service connections."""
    subscription_id: str = ""
    resource_group: str = ""
    acr_name: str = ""
    aml_workspace: str = ""
    aml_compute_target: str = ""
    discovered: List[str] = []   # profile key names that were auto-discovered
    errors: List[str] = []       # non-fatal errors during discovery (shown to user)


# ── effective config ──────────────────────────────────────────────────────────

def _effective(profile_overrides: dict | None) -> dict:
    """Merge profile overrides with global settings into a single flat config dict."""
    s = get_settings()
    ov = profile_overrides or {}
    return {
        "subscription_id": ov.get("azure_subscription_id") or s.azure_subscription_id or "",
        "resource_group":  ov.get("azure_resource_group") or s.aml_resource_group or "",
        "acr_name":        ov.get("acr_name") or s.acr_name or "",
        "aml_workspace":   ov.get("aml_workspace") or s.aml_workspace or "",
        "aml_compute":     ov.get("aml_compute_target") or s.aml_compute_target or "",
    }


# ── individual checks ──────────────────────────────────────────────────────────

def _check_resource_group(cfg: dict) -> PrereqItem:
    name = "Resource Group"
    sub  = cfg["subscription_id"]
    rg   = cfg["resource_group"]

    if not sub:
        return PrereqItem(name=name, status="missing_config",
                          detail="AZURE_SUBSCRIPTION_ID not set",
                          required_for="All Azure resources",
                          fix="Click ⚡ Auto-discover below — or add AZURE_SUBSCRIPTION_ID=<id> to .env")
    if not rg:
        return PrereqItem(name=name, status="missing_config",
                          detail="AZURE_RESOURCE_GROUP not set",
                          required_for="All Azure resources",
                          fix="Click ⚡ Auto-discover below — or add AZURE_RESOURCE_GROUP=<name> to .env  ·  Create: az group create --name <name> --location eastus")
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.resource import ResourceManagementClient
        client = ResourceManagementClient(DefaultAzureCredential(), sub)
        rg_obj = client.resource_groups.get(rg)
        return PrereqItem(name=f"Resource Group ({rg})", status="ok",
                          detail=f"location={rg_obj.location}", required_for="All Azure resources")
    except ImportError:
        return PrereqItem(name=f"Resource Group ({rg})", status="ok",
                          detail="configured (azure-mgmt-resource not installed — skipped live check)",
                          required_for="All Azure resources")
    except Exception as exc:
        err = str(exc)
        st = "not_found" if "ResourceGroupNotFound" in err or "does not exist" in err else "error"
        return PrereqItem(name=f"Resource Group ({rg})", status=st,
                          detail=err[:200], required_for="All Azure resources",
                          fix=f"Create it: az group create --name {rg} --location eastus")


def _check_acr(cfg: dict) -> PrereqItem:
    name = "Container Registry (ACR)"
    sub  = cfg["subscription_id"]
    rg   = cfg["resource_group"]
    acr  = cfg["acr_name"]

    if not acr:
        return PrereqItem(name=name, status="missing_config",
                          detail="ACR_NAME not set",
                          required_for="CI pipeline: docker push → AML environment creation",
                          fix="Click ⚡ Auto-discover below — or add ACR_NAME=<name> to .env  ·  Create: az acr create --resource-group <rg> --name <name> --sku Basic")
    if not sub or not rg:
        return PrereqItem(name=f"ACR ({acr})", status="missing_config",
                          detail="also set AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP",
                          required_for="CI pipeline: docker push → AML environment creation",
                          fix="Click ⚡ Auto-discover below — or add AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP to .env first")
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.containerregistry import ContainerRegistryManagementClient
        client = ContainerRegistryManagementClient(DefaultAzureCredential(), sub)
        acr_obj = client.registries.get(rg, acr)
        return PrereqItem(name=f"ACR ({acr})", status="ok",
                          detail=f"login server: {acr_obj.login_server}",
                          required_for="CI pipeline: docker push → AML environment creation")
    except ImportError:
        return PrereqItem(name=f"ACR ({acr})", status="ok",
                          detail="configured (azure-mgmt-containerregistry not installed — skipped live check)",
                          required_for="CI pipeline: docker push → AML environment creation")
    except Exception as exc:
        err = str(exc)
        st = "not_found" if "ResourceNotFound" in err or "not found" in err.lower() else "error"
        return PrereqItem(name=f"ACR ({acr})", status=st, detail=err[:200],
                          required_for="CI pipeline: docker push → AML environment creation",
                          fix=f"Create it: az acr create --resource-group {rg} --name {acr} --sku Basic")


def _check_aml_workspace(cfg: dict) -> PrereqItem:
    name = "AML Workspace"
    ws   = cfg["aml_workspace"]
    rg   = cfg["resource_group"]

    if not ws:
        return PrereqItem(name=name, status="missing_config",
                          detail="AZURE_ML_WORKSPACE not set",
                          required_for="Training, batch deployment, monitoring pipelines",
                          fix="Click ⚡ Auto-discover below — or add AZURE_ML_WORKSPACE=<name> to .env  ·  Create: az ml workspace create --name <name> --resource-group <rg>")
    try:
        from src.tools.aml_tools import _ml_client
        client = _ml_client()
        ws_obj = client.workspaces.get(ws)
        return PrereqItem(name=f"AML Workspace ({ws})", status="ok",
                          detail=f"location={ws_obj.location}",
                          required_for="Training, batch deployment, monitoring pipelines")
    except Exception as exc:
        err = str(exc)
        st = "not_found" if "not found" in err.lower() or "WorkspaceNotFound" in err else "error"
        return PrereqItem(name=f"AML Workspace ({ws})", status=st, detail=err[:200],
                          required_for="Training, batch deployment, monitoring pipelines",
                          fix=f"Create it: az ml workspace create --name {ws} --resource-group {rg or '<rg>'}  ·  Then restart the backend")


def _check_aml_compute(cfg: dict) -> PrereqItem:
    name = "AML Compute Cluster"
    ct   = cfg["aml_compute"]
    ws   = cfg["aml_workspace"]
    rg   = cfg["resource_group"]

    if not ct:
        return PrereqItem(name=name, status="missing_config",
                          detail="AZURE_ML_COMPUTE_TARGET not set",
                          required_for="Training and batch inference AML jobs",
                          fix="Click ⚡ Auto-discover below — or add AZURE_ML_COMPUTE_TARGET=<cluster-name> to .env  ·  Create via AML Studio → Compute → Compute clusters → New")
    if not ws:
        return PrereqItem(name=f"Compute ({ct})", status="missing_config",
                          detail="also set AZURE_ML_WORKSPACE",
                          required_for="Training and batch inference AML jobs",
                          fix="Set AZURE_ML_WORKSPACE first, then restart the backend")
    try:
        from src.tools.aml_tools import _ml_client
        client = _ml_client()
        compute = client.compute.get(ct)
        state = getattr(compute, "provisioning_state", "unknown")
        return PrereqItem(name=f"Compute Cluster ({ct})", status="ok",
                          detail=f"state={state}",
                          required_for="Training and batch inference AML jobs")
    except Exception as exc:
        err = str(exc)
        st = "not_found" if "not found" in err.lower() or "ComputeNotFound" in err else "error"
        return PrereqItem(name=f"Compute Cluster ({ct})", status=st, detail=err[:200],
                          required_for="Training and batch inference AML jobs",
                          fix=f"Create it in AML Studio → Compute → Compute clusters → New  ·  Or: az ml compute create --name {ct} --type AmlCompute --resource-group {rg or '<rg>'} --workspace-name {ws}")


def _check_azdo_service_connections() -> List[PrereqItem]:
    items: List[PrereqItem] = []
    try:
        import requests
        from src.tools.azdo_tools import _base_url, _headers
        resp = requests.get(
            f"{_base_url()}/_apis/serviceendpoint/endpoints?api-version=7.1",
            headers=_headers(), timeout=30,
        )
        if resp.status_code >= 300:
            items.append(PrereqItem(
                name="AzDO Service Connections",
                status="error",
                detail=f"API {resp.status_code}: {resp.text[:200]}",
                required_for="All pipelines connecting to Azure",
            ))
            return items

        endpoints = resp.json().get("value", [])
        types_names = [(e.get("type", "").lower(), e.get("name", ""), e.get("data", {}))
                       for e in endpoints]

        arm = next(((n, d) for t, n, d in types_names
                    if "azurerm" in t or ("arm" in t and "azure" in t)), None)
        if arm:
            items.append(PrereqItem(name=f"AzDO → Azure RM ({arm[0]})", status="ok",
                                    required_for="CD pipelines: AML job submission, endpoint deployment"))
        else:
            items.append(PrereqItem(
                name="AzDO → Azure RM service connection",
                status="not_found",
                detail="No Azure Resource Manager service connection found",
                required_for="CD pipelines: AML job submission, endpoint deployment",
                fix="AzDO Project Settings → Service Connections → New service connection → Azure Resource Manager → Service Principal (automatic) → Grant Contributor on your resource group",
            ))

        # Docker Registry service connection intentionally omitted:
        # generated CI pipelines use "az acr build" via the ARM connection above,
        # so no separate Docker Registry connection is required.

    except Exception as exc:
        items.append(PrereqItem(
            name="AzDO Service Connections",
            status="error",
            detail=str(exc)[:200],
            required_for="All pipelines connecting to Azure",
        ))
    return items


# ── auto-discovery ────────────────────────────────────────────────────────────

def _arm_get(token: str, url: str) -> dict:
    """GET an Azure Resource Manager REST endpoint with the given bearer token."""
    import requests as req
    resp = req.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def auto_discover_from_azdo() -> DiscoveredConfig:
    """Discover Azure config from AzDO service connections + DefaultAzureCredential.

    Uses Azure Resource Manager REST API directly — no management SDK packages
    required beyond azure-identity (which is already a project dependency).

    Steps:
      1. Query AzDO service connections → ARM connection → subscriptionId + resourceGroupName
      2. Obtain an ARM bearer token via DefaultAzureCredential (requires az login)
      3. List AML workspaces in the subscription (scoped to RG if available)
      4. List ACR registries in the subscription (scoped to RG if available)
      5. List AmlCompute clusters in the discovered workspace

    All steps are best-effort; partial results are returned when credentials are
    missing or resources do not exist. Errors are collected in result.errors.
    """
    import requests as req

    result = DiscoveredConfig()
    ARM = "https://management.azure.com"

    # ── Step 1: AzDO ARM service connection → subscription + RG ──────────────
    try:
        from src.tools.azdo_tools import _base_url, _headers
        resp = req.get(
            f"{_base_url()}/_apis/serviceendpoint/endpoints?api-version=7.1",
            headers=_headers(), timeout=30,
        )
        if resp.status_code == 401:
            result.errors.append(
                "AzDO auth failed — ensure the PAT has 'Service Connections (Read)' scope"
            )
        elif resp.status_code >= 300:
            result.errors.append(f"AzDO service connections API: HTTP {resp.status_code}")
        else:
            for ep in resp.json().get("value", []):
                if "azurerm" in ep.get("type", "").lower():
                    data = ep.get("data", {})
                    if not result.subscription_id and data.get("subscriptionId"):
                        result.subscription_id = data["subscriptionId"]
                        result.discovered.append("azure_subscription_id")
                    if not result.resource_group and data.get("resourceGroupName"):
                        result.resource_group = data["resourceGroupName"]
                        result.discovered.append("azure_resource_group")
                    break
    except Exception as exc:
        result.errors.append(f"AzDO service connections: {exc}")

    if not result.subscription_id:
        result.errors.append(
            "Subscription ID not found in AzDO ARM service connection. "
            "Run 'az login' locally, then re-check your ARM connection includes a subscription."
        )
        return result

    # ── Step 2: acquire ARM bearer token ─────────────────────────────────────
    token = ""
    try:
        from azure.identity import DefaultAzureCredential
        token = DefaultAzureCredential().get_token(f"{ARM}/.default").token
    except ImportError:
        result.errors.append("azure-identity not installed — cannot authenticate to Azure")
        return result
    except Exception as exc:
        result.errors.append(
            f"Azure auth failed ({str(exc)[:150]}). "
            "Run 'az login' on this machine so DefaultAzureCredential can authenticate."
        )
        return result

    sub = result.subscription_id
    rg  = result.resource_group

    # ── Step 3: list AML workspaces ───────────────────────────────────────────
    try:
        if rg:
            url = f"{ARM}/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.MachineLearningServices/workspaces?api-version=2023-04-01"
        else:
            url = f"{ARM}/subscriptions/{sub}/providers/Microsoft.MachineLearningServices/workspaces?api-version=2023-04-01"
        workspaces = _arm_get(token, url).get("value", [])
        if workspaces:
            ws = workspaces[0]
            result.aml_workspace = ws["name"]
            result.discovered.append("aml_workspace")
            # derive RG from resource ID if not already known
            if not rg:
                rid_parts = ws.get("id", "").split("/")
                try:
                    result.resource_group = rid_parts[rid_parts.index("resourceGroups") + 1]
                    result.discovered.append("azure_resource_group")
                    rg = result.resource_group
                except (ValueError, IndexError):
                    pass
        else:
            result.errors.append("No AML workspaces found in the subscription/resource group")
    except Exception as exc:
        result.errors.append(f"AML workspace list: {str(exc)[:200]}")

    # ── Step 4: list ACR registries ───────────────────────────────────────────
    try:
        if rg:
            url = f"{ARM}/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.ContainerRegistry/registries?api-version=2023-01-01-preview"
        else:
            url = f"{ARM}/subscriptions/{sub}/providers/Microsoft.ContainerRegistry/registries?api-version=2023-01-01-preview"
        registries = _arm_get(token, url).get("value", [])
        if registries:
            result.acr_name = registries[0]["name"]
            result.discovered.append("acr_name")
        else:
            result.errors.append("No ACR registries found — create one with: az acr create --resource-group <rg> --name <name> --sku Basic")
    except Exception as exc:
        result.errors.append(f"ACR list: {str(exc)[:200]}")

    # ── Step 5: list AmlCompute clusters in workspace ─────────────────────────
    if result.aml_workspace and rg:
        try:
            url = (
                f"{ARM}/subscriptions/{sub}/resourceGroups/{rg}"
                f"/providers/Microsoft.MachineLearningServices/workspaces/{result.aml_workspace}"
                f"/computes?api-version=2023-04-01"
            )
            computes = _arm_get(token, url).get("value", [])
            aml_computes = [
                c["name"] for c in computes
                if c.get("properties", {}).get("computeType", "").lower() == "amlcompute"
            ]
            if aml_computes:
                result.aml_compute_target = aml_computes[0]
                result.discovered.append("aml_compute_target")
            else:
                result.errors.append("No AmlCompute clusters found in workspace — create one in AML Studio → Compute → Compute clusters")
        except Exception as exc:
            result.errors.append(f"AML compute list: {str(exc)[:200]}")

    return result


# ── public API ─────────────────────────────────────────────────────────────────

def check_all_prerequisites(profile_overrides: dict | None = None) -> InfraReport:
    """Run all six prerequisite checks.

    profile_overrides values take priority over global .env settings, allowing
    per-project configuration without editing .env.
    """
    cfg = _effective(profile_overrides)
    checks: List[PrereqItem] = [
        _check_resource_group(cfg),
        _check_acr(cfg),
        _check_aml_workspace(cfg),
        _check_aml_compute(cfg),
    ]
    checks.extend(_check_azdo_service_connections())
    # Data-plane paths are prerequisites too: the user must place data in blob and the
    # pipelines reference those paths. Surface them alongside ACR/workspace readiness.
    try:
        from src.tools.data_path_tools import verify_data_paths
        checks.extend(verify_data_paths(profile_overrides or {}).checks)
    except Exception:
        pass
    return InfraReport(checks=checks)
