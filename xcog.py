import fsspec
import os
import io
import pandas as pd
import xarray as xr

import rasterio.warp
import shapely.geometry
import pystac
import datetime

from typing import Optional, Mapping


def name_block(
    block: xr.DataArray,
    prefix: str = "",
    include_band: bool = True,
    x_dim="x",
    y_dim="y",
):
    """
    Get the name for a block, based on the coordinates at the top-left corner.
    """
    time = pd.Timestamp(block.coords["time"][0].item()).isoformat()

    x = y = ""
    if x_dim:
        x = block.coords[x_dim][0].item()
    if y_dim:
        y = block.coords[y_dim][0].item()

    prefix = prefix or ""

    if include_band:
        band = block.coords["band"][0].item()
        band = f"band={band}-"
    else:
        band = ""

    if x:
        x = f"x={x}"
    if y:
        y = f"y={y}"

    if not any([band, x, y]):
        raise ValueError("Must specify `include_band` or `x_dim` or `y_dim`")

    name = "-".join([thing for thing in [band, y, x] if thing])

    blob_name = os.path.join(prefix, f"time={time}", f"{name}.tif")
    return blob_name


def write_block(
    block: xr.DataArray,
    prefix: str = "",
    x_dim: str = "x",
    y_dim: str = "y",
    storage_options: Optional[Mapping[str, str]] = None,
):
    # this is specific to azure blob storage. We could generalize to accept an fsspec URL.
    import rioxarray  # noqa

    storage_options = storage_options or {}
    blob_name = name_block(block, prefix=prefix, x_dim=x_dim, y_dim=y_dim)

    fs, _, paths = fsspec.get_fs_token_paths(blob_name, storage_options=storage_options)
    if len(paths) > 1:
        raise ValueError("too many paths", paths)
    path = paths[0]
    memfs = fsspec.filesystem("memory")

    with memfs.open("data", "wb") as buffer:
        block.squeeze().rio.to_raster(buffer, driver="COG")
        buffer.seek(0)
        fs.pipe_file(path, buffer)

    result = (
        block.isel(**{k: slice(1) for k in block.dims}).astype(object).compute().copy()
    )
    template_item = pystac.Item("id", None, None, datetime.datetime(2000, 1, 1), {})
    item = itemize(block, template_item, x_dim=x_dim, y_dim=y_dim, prefix=prefix)

    result[(0,) * block.ndim] = item
    return result


def itemize(
    block,
    item: pystac.Item,
    time_dim="time",
    x_dim="x",
    y_dim="y",
    prefix: str = "",
    href=name_block,
):
    import rioxarray  # noqa

    item = item.clone()
    dst_crs = rasterio.crs.CRS.from_epsg(4326)

    bbox = rasterio.warp.transform_bounds(block.rio.crs, dst_crs, *block.rio.bounds())
    geometry = shapely.geometry.mapping(shapely.geometry.box(*bbox))

    item.id = name_block(block, include_band=False, x_dim=x_dim, y_dim=y_dim)
    item.geometry = geometry
    item.bbox = bbox
    item.datetime = pd.Timestamp(block.coords[time_dim].item()).to_pydatetime()

    ext = pystac.extensions.projection.ProjectionExtension.ext(
        item, add_if_missing=True
    )
    ext.bbox = block.rio.bounds()
    ext.shape = block.shape[-2:]
    ext.epsg = block.rio.crs.to_epsg()
    ext.geometry = shapely.geometry.mapping(shapely.geometry.box(*ext.bbox))
    ext.transform = list(block.rio.transform())[:6]
    ext.add_to(item)

    asset = pystac.Asset(
        href=name_block(block, x_dim=x_dim, y_dim=y_dim, prefix=prefix),
        media_type=pystac.MediaType.COG,
    )
    asset.extra_fields["eo:bands"] = [
        dict(
            name=block.band.item(),
            common_name=block.common_name.item(),
            center_wavelength=block.center_wavelength.item(),
            full_width_half_max=block.full_width_half_max.item(),
        )
    ]

    item.add_asset(block.band[0].item(), asset)

    return item
